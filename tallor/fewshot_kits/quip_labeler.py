#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#

from collections import defaultdict
import os
from copy import deepcopy
# from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import datetime
import logging
import torch
import torch.utils.data as data
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast
from datasets import load_metric

from tallor.data_loader import FewShotPredDataset
from tallor.utils import LabelInstance

# np.set_printoptions(threshold=sys.maxsize)


# The following snippet is from FewNERD
#    (https://github.com/thunlp/Few-NERD/)
# This source code is licensed under the Apache 2.0 license,
# found in the 3rd-party-licenses.txt file in the root directory of this source tree.
def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0



class QuipLabeler:

    def __init__(self, label_id_mapper, unlabel_data, 
                 val_data_loader, test_data_loader, batch_size, 
                 max_length=60, beta=0.2, beta_prime=0.05, 
                 mode='cotrain', ignore_index=-100, exp_dir='.'):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.beta = beta
        self.beta_prime = beta_prime
        self.label_id_mapper = label_id_mapper
        self.batch_size = batch_size
        self.test_data_loader = test_data_loader
        self.max_length = max_length
        self.mode = mode
        self.logger = set_logger(exp_dir, str(datetime.datetime.now()))
        self.high_precision_instances = {}
        self.negative_instance_set = set()
        self.negative_instances = {}
        self.to_label_instance_set = set()
        self.to_label_instances = {}
        self.candidate_instances = defaultdict(set)
        self.unlabel_data = unlabel_data
        self.ignore_index = ignore_index
        self.train_data_loader = None
        self.labeling_data_loader = None

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
                            'roberta-large',
                            use_fast=True,
                            # use_auth_token=True,
                            add_prefix_space=True)
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
                "requirement"
            )
        labeling_dataset = FewShotPredDataset(deepcopy(self.unlabel_data), self.tokenizer, self.label_id_mapper, max_length=self.max_length)
        self.labeling_dataset = list(map(self.tokenize_and_align_labels, labeling_dataset.samples))

        val_dataset = val_data_loader._dataset.training_data
        val_dataset = FewShotPredDataset(deepcopy(val_dataset), self.tokenizer, self.label_id_mapper, max_length=self.max_length)
        val_dataset = list(map(self.tokenize_and_align_labels, val_dataset.samples))
        self.val_dataset = data.DataLoader(dataset=val_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=8,
                                                collate_fn=self.collate_fn)
        self.val_data_loader = iter(self.val_dataset)

        test_dataset = test_data_loader._dataset.training_data
        test_dataset = FewShotPredDataset(deepcopy(test_dataset), self.tokenizer, self.label_id_mapper, max_length=self.max_length)
        test_dataset = list(map(self.tokenize_and_align_labels, test_dataset.samples))
        self.test_dataset = data.DataLoader(dataset=test_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=8,
                                                collate_fn=self.collate_fn)
        self.test_data_loader = iter(self.test_dataset)

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)


    def get_three_sets(self):

        return self.high_precision_instances, self.negative_instance_set, self.to_label_instance_set

    def set_three_sets(self, high_precision_instances, negative_instance_set, to_label_instance_set):

        del self.high_precision_instances
        del self.negative_instance_set
        del self.to_label_instance_set

        self.high_precision_instances = defaultdict(set)
        self.negative_instance_set = set()
        self.to_label_instance_set = set()

        for k, v in high_precision_instances.items():
            for instance in v:
                self.high_precision_instances[k].add(instance)

        for instance in negative_instance_set:
            self.negative_instance_set.add(instance)
        
        for instance in to_label_instance_set:
            self.to_label_instance_set.add(instance)

        del self.negative_instances
        del self.to_label_instances

        self.negative_instances = {}
        for instance in negative_instance_set:
            self.negative_instances[(instance.data_idx, instance.span[0], instance.span[1])] = instance

        self.to_label_instances = {}
        for instance in to_label_instance_set:
            self.to_label_instances[(instance.data_idx, instance.span[0], instance.span[1])] = instance
    
    def set_three(self, high_precision_instances, negative_instance_set, to_label_instance_set):

        del self.high_precision_instances
        del self.negative_instance_set
        del self.to_label_instance_set

        self.high_precision_instances = defaultdict(set)
        self.negative_instance_set = set()
        self.to_label_instance_set = set()

        for instance in high_precision_instances:
            self.high_precision_instances[instance.label].add(instance)

        for instance in negative_instance_set:
            self.negative_instance_set.add(instance)
        
        for instance in to_label_instance_set:
            self.to_label_instance_set.add(instance)

        del self.negative_instances
        del self.to_label_instances

        self.negative_instances = {}
        for instance in negative_instance_set:
            self.negative_instances[(instance.data_idx, instance.span[0], instance.span[1])] = instance

        self.to_label_instances = {}
        for instance in to_label_instance_set:
            self.to_label_instances[(instance.data_idx, instance.span[0], instance.span[1])] = instance

    
    def tokenize_and_align_labels(self, instance):

        if isinstance(instance, LabelInstance):
            tokenized_inputs = self.tokenizer(
                instance.sentence,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            tokenized_inputs["data_idx"] = instance.data_idx
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            assert not isinstance(instance.ground_truth_tag, list)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    # if we are dealing with the high_precision_instance_set, we should restrict predictions
                    # to be within the span mentioned. Anything outside that is off limits since we do not know the
                    # actual tag of the rest of the sentence.
                    if word_idx in list(range(instance.span[0], instance.span[1]+1)):
                        label_ids.append(self.label_id_mapper.label2id[instance.ground_truth_tag])
                    else:
                        # label_ids.append(self.label_id_mapper.get_neg_id())
                        label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # if word_idx in list(range(instance.span[0], instance.span[1]+1)):
                    #     label_ids.append(self.label_id_mapper.label2id[instance.ground_truth_tag])
                    # else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
        else:
            tokenized_inputs = self.tokenizer(
                instance.words,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            tokenized_inputs["data_idx"] = [instance.data_idx[0]] * self.max_length
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    idx = instance.tags[word_idx] if instance.tags[word_idx] != 'O' else ''
                    label_ids.append(self.label_id_mapper.label2id[idx])
                # For the other tokens in a word, we set the label to -100
                else:
                    # idx = instance.tags[word_idx] if instance.tags[word_idx] != 'O' else ''
                    # label_ids.append(self.label_id_mapper.label2id[idx])
                    label_ids.append(-100)
                previous_word_idx = word_idx
        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs
    
    def collate_fn(self, batch):

        dict_batch = {}

        for datapoint in batch:
            for (k, v) in datapoint.items():
                if k in dict_batch:
                    dict_batch[k].append(v)
                else:
                    dict_batch[k] = [v]

        return dict_batch

    def reset_train_dataset_and_loader(self):

        if self.train_data_loader:
            del self.train_data_loader

        high_precision_instance_set = self.get_high_precision_set()

        train_dataset = list(map(self.tokenize_and_align_labels, high_precision_instance_set))
        self.train_dataset = data.DataLoader(dataset=train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=8,
                                                collate_fn=self.collate_fn)
        self.train_data_loader = iter(self.train_dataset)


    def reset_labeling_dataset_and_loader(self):

        if self.labeling_data_loader:
            del self.labeling_data_loader

        self.labeling_data_loader = data.DataLoader(dataset=self.labeling_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=8,
                                                    collate_fn=self.collate_fn)
        self.labeling_data_loader = iter(self.labeling_data_loader)


    def get_high_precision_set(self, high_precision_instances=None):
        '''
        positive set is the embeddings of high precision set
        '''
        if high_precision_instances is None: high_precision_instances = self.high_precision_instances
        high_precision_set = set()
        for label, instance_set in high_precision_instances.items():
            high_precision_set |= instance_set

        return high_precision_set
    
    def compute_metrics(self, p):

        predictions, labels = p
        # predictions = np.argmax(predictions, axis=2)
        label_list = self.label_id_mapper.all_labels()

        # Remove ignored index (special tokens)
        true_predictions = [
            label_list[p] for (p, l) in zip(predictions, labels) if l != -100
        ]
        true_labels = [
            label_list[l] for (_, l) in zip(predictions, labels) if l != -100
        ]
        true_predictions = [pred if pred!='' else 'O' for pred in true_predictions]
        true_labels = [label if label!='' else 'O' for label in true_labels]

        metric = load_metric("seqeval")
        results = metric.compute(predictions=[true_predictions], references=[true_labels])
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def update_three_set(self, positive_list):

        # high_precision_instance_set = defaultdict(set)
        for instance in positive_list:

            self.high_precision_instances[instance.label].add(instance)
            if instance in self.negative_instance_set:
                self.negative_instance_set.remove(instance)
            if instance in self.to_label_instance_set:
                self.to_label_instance_set.remove(instance)
        
        # return high_precision_instance_set


    def pipeline(self,
                 model,
                 model_name,
                 step=0,
                 rule_labeled_set=set(),
                 train_iter=12000,
                 warmup_step=0):


        print('Begin training QuIP!')
        self.reset_train_dataset_and_loader()
        self.train(model, model_name, train_iter=train_iter, warmup_step=warmup_step)
        print('QuIP training done!')

        print('Begin labeling using QuIP!')
        self.reset_labeling_dataset_and_loader()
        label_instances_dict = self.labeling(model, rule_labeled_set)
        print('QuIP labeling done!')

        self.select_instance_and_update_three_sets(label_instances_dict, step)
        positive_instances = self.get_high_precision_set()

        print(f'STEP:{step}: After QuIP labeling, we get {len(positive_instances)} positive instances, {len(self.negative_instance_set)} negative instances, {len(self.to_label_instance_set)} unlabel instances.')
        return positive_instances

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    # The following snippet is derived from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def train(self,
              model,
              model_name,
              learning_rate=2e-5,
              train_iter=12000,
              warmup_step=0):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        '''

        print("Start training...")
    
        # Init optimizer
        parameters_to_optimize = list(model.parameters())
        optimizer = AdamW(parameters_to_optimize, lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        model = model.cuda()
        model.train()

        # Training
        iter_loss = 0.0
        iter_sample = 0
        predictions = []
        labels = []
        for it in range(train_iter):
            instances = next(self.train_data_loader, None)
            if instances is None:
                self.train_data_loader = iter(self.train_dataset)
                instances = next(self.train_data_loader)
            instances = {key:torch.tensor(value).long().cuda() for key, value in instances.items()}
            logits, pred = model(instances)
            loss = model.loss(logits, instances['labels'])
            predictions.append(pred.cpu().numpy())
            labels.append(instances['labels'].cpu().numpy())
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            iter_loss += self.item(loss.data)
            iter_sample += 1
            if (it + 1) % 100 == 0:
                cum_predictions = np.concatenate([np.concatenate(prediction, axis=0) for prediction in predictions], axis=0)
                cum_labels = np.concatenate([np.concatenate(label, axis=0) for label in labels], axis=0)
                pred_cnt, label_cnt, correct = model.metrics_by_entity(cum_predictions, cum_labels)
                precision = correct / pred_cnt
                recall = correct / label_cnt
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                sys.stdout.write('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                    .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
            sys.stdout.flush()
        
        print("\n####################")
        print("Finish training " + model_name)

    # The following snippet is derived from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def eval(self,
            model,
            eval_iter,
            ckpt=None): 

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = iter(self.val_dataset)
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        pred_cnt = 1e-6
        label_cnt = 1e-6
        correct_cnt = 1e-10

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct
        with torch.no_grad():
            for it in range(eval_iter):
                instances = next(eval_dataset, None)
                if instances is None: break
                instances = {key:torch.tensor(value).long().cuda() for key, value in instances.items()}
                label = instances['labels']
                _, pred = model(instances)

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

            precision = correct_cnt / pred_cnt
            recall = correct_cnt /label_cnt
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\n')
            sys.stdout.flush()
            
        return f1, precision, recall


    def labeling(self,
            model,
            rule_labeled_set): 


        model.eval()

        # eval_data_loader = self.test_data_loader

        pred_cnt = 1e-6
        label_cnt = 1e-6
        correct_cnt = 1e-10

        predictions = []
        labels = []
        label_instances_dict = defaultdict(list)

        with torch.no_grad():

            it = 0

            print("Start labeling...")

            unlabel_data_copy = deepcopy(self.unlabel_data)
            instance = next(self.labeling_data_loader, None)
            while instance:
                instance = {key:torch.tensor(value).long().cuda() for key, value in instance.items()}
                nearest_dist, pred = model.predict(instance)
                label = instance['labels']
                data_idx = instance['data_idx']
                label_instances_dict = self.update_prediction_result(label_instances_dict, unlabel_data_copy, rule_labeled_set, label, pred, nearest_dist, data_idx)

                predictions.append(pred.cpu().numpy())
                labels.append(instance['labels'].cpu().numpy())

                instance = next(self.labeling_data_loader, None)

            del unlabel_data_copy
            
            cum_predictions = np.concatenate([np.concatenate(prediction, axis=0) for prediction in predictions], axis=0)
            cum_labels = np.concatenate([np.concatenate(label, axis=0) for label in labels], axis=0)
            pred_cnt, label_cnt, correct = model.metrics_by_entity(cum_predictions, cum_labels)
            pred_cnt = max(1, pred_cnt)
            precision = correct / pred_cnt 
            label_cnt = max(1, label_cnt)
            recall = correct / label_cnt
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            sys.stdout.write('[LABELING] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
            sys.stdout.flush()
        return label_instances_dict # , precision, recall, f1, fp_error, fn_error, within_error, outer_error

    def update_prediction_result(self, label_instances_dict, unlabel_datapoints, rule_labeled_set, label, pred, nearest_dist, data_id):

        new_pred = pred[label!=self.ignore_index]
        new_data_id = data_id[label!=self.ignore_index]
        new_nearest_dist = nearest_dist[label!=self.ignore_index]
        data_id_list = torch.unique(new_data_id).tolist()
        neg_id = self.label_id_mapper.get_neg_id()

        assert new_pred.size() == new_data_id.size()

        for data_idx in data_id_list:

            spans = []
            pred_labels = []
            dist = []

            sentence_pred = new_pred[new_data_id==data_idx].tolist()
            nd = new_nearest_dist[new_data_id==data_idx].tolist()
            span_left = 0

            if len(unlabel_datapoints[data_idx].sentence) == len(sentence_pred):
                while span_left < len(sentence_pred):
                    if sentence_pred[span_left] == neg_id:
                        span_left += 1
                        continue
                    else:
                        d = 0
                        span_right = span_left
                        while span_right < len(sentence_pred) and sentence_pred[span_right] == sentence_pred[span_left]:
                            d += nd[span_right]
                            span_right += 1
                        # dist.append(d / (span_right-span_left))
                        spans.append((span_left, span_right-1))
                        pred_labels.append(sentence_pred[span_left])
                        span_left = span_right
                        span_right -= 1

                sentence_pred_with_ignored = pred[data_id==data_idx].tolist()
                new_label = label[data_id==data_idx]
                nd_with_ignored = nearest_dist[data_id==data_idx].tolist()
                assert len(sentence_pred_with_ignored) == len(new_label), print(2, len(sentence_pred_with_ignored), len(new_label))

                span_left = 0

                while span_left < len(sentence_pred_with_ignored):
                    if sentence_pred_with_ignored[span_left] != neg_id and new_label[span_left] != self.ignore_index:
                        d = 0
                        span_right = span_left
                        while span_right < len(sentence_pred_with_ignored) and \
                            (sentence_pred_with_ignored[span_right] == sentence_pred_with_ignored[span_left] or new_label[span_right] == self.ignore_index):
                            d += nd_with_ignored[span_right]
                            span_right += 1
                        dist.append(d / (span_right-span_left))
                        span_left = span_right
                        span_right -= 1
                    else:
                        span_left += 1
                        continue

                assert len(spans) == len(dist), print(unlabel_datapoints[data_idx].sentence, sentence_pred, sentence_pred_with_ignored)

                if pred_labels:
                    datapoint = unlabel_datapoints[data_idx]
                    span2id = {span:idx for idx, span in enumerate(datapoint.spans)}
                    for span_idx, span in enumerate(spans):
                        start, end = span
                        if span in span2id:
                            fewshot_label = pred_labels[span_idx]
                            # pred_tag = self.labeling_dataset.label2tag[fewshot_label]
                            pred_label = fewshot_label # self.label_id_mapper.get_id(pred_tag)
                            soft_label = self.label_id_mapper.get_soft_label(pred_label)
                            labeled_instance = None
                            if (data_idx, start, end) in self.negative_instances:
                                instance = self.negative_instances[(data_idx, start, end)]
                                if instance not in rule_labeled_set:
                                    if instance.parent==None or instance.parent==True:
                                        instance.label = pred_label
                                        instance.label_prob = soft_label
                                        labeled_instance = instance
                                    else:
                                        parent = instance.parent
                                        parent.label = pred_label
                                        parent.label_prob = soft_label
                                        labeled_instance = parent
                            elif (data_idx, start, end) in self.to_label_instances:
                                instance = self.to_label_instances[(data_idx, start, end)]
                                if instance not in rule_labeled_set:
                                    if instance.parent==None or instance.parent==True:
                                        instance.label = pred_label
                                        instance.label_prob = soft_label
                                        labeled_instance = instance
                                    else:
                                        parent = instance.parent
                                        parent.label = pred_label
                                        parent.label_prob = soft_label
                                        labeled_instance = parent
                            if labeled_instance:
                                label_instances_dict[pred_label].append((labeled_instance, dist[span_idx]))

        return label_instances_dict


    def select_instance_and_update_three_sets(self, label_instances_dict, step):

        # high_precision_instance_set = defaultdict(set)
        for label, instances_with_dist in label_instances_dict.items():
            print_count = 0
            # instances_with_dist.extend(list(self.candidate_instances[label]))
            instances_with_dist = sorted(instances_with_dist, key=lambda x: x[1], reverse=True)
            for idx, (instance, dist) in enumerate(instances_with_dist):
                if idx > len(instances_with_dist) * (self.beta + step * self.beta_prime):
                    if instance in self.negative_instance_set:
                        # self.candidate_instances[label].add((instance, dist))
                        self.negative_instance_set.remove(instance)
                        self.to_label_instance_set.add(instance)
                else:
                    if instance not in self.high_precision_instances[label]:
                        self.high_precision_instances[label].add(instance)
                        # high_precision_instance_set[label].add(instance)
                        if print_count < 10:
                            print('QuIP labeled: ', instance.data_idx, self.label_id_mapper.get_label(instance.label), instance.ground_truth_tag, instance.span, instance.sentence[instance.span[0]:instance.span[1]+1], dist, instance.sentence)
                        print_count += 1

                        if instance in self.negative_instance_set:
                            self.negative_instance_set.remove(instance)
                        
                        if instance in self.to_label_instance_set:
                            self.to_label_instance_set.remove(instance)


def set_logger(exp_dir, timestamp):
    ## set logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('FewshotLabeler')
    
    file_handler = logging.FileHandler(f'{exp_dir}/logging/FewshotLabeler-{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

    return logger