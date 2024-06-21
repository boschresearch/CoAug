#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#

import os
import numpy as np
import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizerFast
from transformers import PreTrainedTokenizerFast
# from transformers import Trainer, DataCollatorForTokenClassification, TrainingArguments
from transformers import logging
logging.set_verbosity_error()

from datasets import load_dataset

from tallor.categorical_accuracy import CategoricalAccuracy
from tallor.precision_recall_f1 import PrecisionRecallF1


TAG_TYPE_TO_QUESTION = {
        "O": "What is a generic object ?",
        # CONLL 2003
        "PER": "Who is a person ?",
        "LOC": "What is a location ?",
        "ORG": "What is an organization ?",
        "MISC": "What is a miscellaneous entity ?",
        # WNUT 17
        "person": "Who is a person ?",
        "location": "What is a location ?",
        "corporation": "What is a corporation ?",
        "product": "What is a product ?",
        "creative-work": "What is a creative work ?",
        "group": "What is a group ?",
        # BC5CDR + NCBI-Disease + CHEMDNER
        "Chemical": "What is a chemical compound ?",
        "Disease": "What is a disease ?",
        # WikiGold
        "organization": "What is an organization ?"
}

def question_to_classifier_head(model_path):

    # load network weights
    model_params = torch.load(model_path)
    input_dim, hidden_dim = model_params['dense.weight'].shape
    num_classes = model_params['out_proj.weight'].shape[1]

    # instantiate network and load the stored params
    model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                          nn.Tanh(),
                          nn.Linear(hidden_dim, num_classes))
    model.load_state_dict(model_params, strict=False)
    return model

@torch.no_grad()
def label_to_classifier(model, tokenizer, mlp, label):

    model.eval()
    q_tokens = tokenizer(TAG_TYPE_TO_QUESTION[label], return_tensors='pt')
    cls_features = model.roberta(**q_tokens).last_hidden_state[:, 0] # we only want the <s> state
    lbl_features = mlp(cls_features)[0] # we only need the embedding
    normed_lbl_features = lbl_features / torch.norm(lbl_features)
    return normed_lbl_features

class QuIP(nn.Module):

    def __init__(self, label_list=None, init=True, ner_label=None):

        super(QuIP, self).__init__()
        
        assert label_list is not None, 'label_list needs to be provided in the input!'

        model_path = 'pretrained_models/quip-hf' if init else 'roberta-large'
        self.num_labels = len(label_list)
        self.config = RobertaConfig.from_pretrained(
                        model_path,
                        num_labels=self.num_labels,
                        )
        self.model = RobertaForTokenClassification.from_pretrained(
                        model_path,
                        config=self.config,
                        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
                            'roberta-large',
                            use_fast=True,
                            add_prefix_space=True)

        ## Initialize the classification heads for the NER task using the QuIP model.
        if init:
            classifier_head = question_to_classifier_head(os.path.join('pretrained_models/quip/model_qa_head_start.pt'))
            self.model.classifier.weight.data = torch.stack(
                [label_to_classifier(self.model, self.tokenizer, classifier_head, label) for label in label_list],
                dim=0
            )

        self.cost = nn.CrossEntropyLoss()

        # self.ner_label = ner_label
        self.ner_neg_id = ner_label.get_neg_id()
        self.ner_acc = CategoricalAccuracy(top_k=1, tie_break=False)
        self.ner_prf = PrecisionRecallF1(neg_label=self.ner_neg_id)
        self.ner_prf_b = PrecisionRecallF1(neg_label=self.ner_neg_id, binary_match=True)
    
    def forward(self, instance):
        '''
        inputs: the input instances for processing with QuIP
        '''
        inputs = {key:value for key, value in instance.items() if key != 'data_idx'}
        outputs = self.model(**inputs)
        logits = outputs.logits
        _, pred = torch.max(logits, dim=-1)

        return logits, pred
    
    def loss(self, logits, label):
        logits = logits.view(-1, self.num_labels)
        label = label.view(-1)
        assert logits.shape[0] == label.shape[0]
        
        return self.cost(logits, label)
    
    def predict(self, instance):
        inputs = {key:value for key, value in instance.items() if key != 'data_idx'}
        outputs = self.model(**inputs)
        logits = outputs.logits
        max_vals, pred = torch.max(logits.softmax(dim=-1), dim=-1)
        return max_vals, pred
    
    def metrics_by_entity(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        # pred = pred.view(-1)
        # label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        pred = pred.tolist()
        label = label.tolist()
        pred_class_span = self.__get_class_span_dict__(pred)
        label_class_span = self.__get_class_span_dict__(label)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt
    
    def __delete_ignore_index(self, pred, label):
        pred = pred[label != -100]
        label = label[label != -100]
        assert pred.shape[0] == label.shape[0]
        return pred, label
    
    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span
    
    def metric_reset(self):

        self.ner_acc.reset()
        self.ner_prf.reset()
        self.ner_prf_b.reset()