#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#  This source code is derived from TALLOR
#    (https://github.com/JiachengLi1995/TALLOR)
#  licensed under the AGPL-3.0 license,
#  cf. 3rd-party-licenses.txt file in the root directory of this source tree.
#
#
#

from collections import defaultdict
import torch
import torch.utils.data as data
import os
import random
import json
import numpy as np
from tallor.utils import DataPoint, list_to_dict
from tallor.fewshot_kits.fewshot_sampler import FewshotSampler
from copy import deepcopy
import spacy
from spacy.tokens import Doc

from tqdm import tqdm

class Parser:

    def __init__(self):

        self.parser = spacy.load('en_core_web_sm')
        self.parser.remove_pipe('ner')
        self.parser.tokenizer = self.custom_tokenizer

    def custom_tokenizer(self, text):
        tokens = text.split(' ')
        return Doc(self.parser.vocab, tokens)

    def parse(self, sentence):

        return self.parser(sentence)

class MissingDict(dict):
    """
    If key isn't there, returns default value. Like defaultdict, but it doesn't store the missing
    keys that were queried.
    """
    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val

def format_label_fields(ner):
    
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start, span_end), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    return ner_dict


class DataSet(data.Dataset):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, root, filename, encoder, batch_size, label_id_mapper, is_train, opt, mode='training'):
        self.batch_size = batch_size
        self.max_span_width = opt.max_span_width
        self.label_id_mapper = label_id_mapper
        self.encoder = encoder
        ## spacy
        self.parser = Parser()  ## use customized parser to ensure that we have the same tokens

        self.label_data = []
        self.unlabel_data = []

        if mode == 'training':

            labeled_ratio = opt.labeled_ratio
            path = os.path.join(root, filename + ".json")

            data = []
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data.append(json.loads(line))
            
            print(f'Begin processing {filename} dataset...')
            
            processed_data = self.preprocess(data)

            data = []
            for line in processed_data:
                sentence, spans, ner_labels, ner_tags = line
                data.append(DataPoint(
                            sentence=sentence,
                            spans=spans,
                            ner_labels=ner_labels,
                            ner_tags=ner_tags,
                            parsed_tokens=self.parser.parse(' '.join(sentence)),
                            label_num = self.label_id_mapper.get_num()
                        ))

            labeled_num = 0
            unlabeled_num = 0
            if not is_train or labeled_ratio==1:
                self.training_data = data
                labeled_num = len(self.training_data)
                unlabeled_num = 0
            else:
                index = list(range(len(data)))
                labeled_index = index[:int(labeled_ratio*len(index))]
                unlabel_index = index[int(labeled_ratio*len(index)):]

                self.label_data = [data[i] for i in labeled_index]

                for i in unlabel_index:

                    data[i].unlabel_reset() ## set all label mask to 0
                    self.unlabel_data.append(data[i])
                    
                self.training_data = deepcopy(self.label_data)

                labeled_num = len(labeled_index)
                unlabeled_num = len(unlabel_index)
                
            print(f'Done. {filename} dataset has {len(data)} instances. \n Among them, we use {labeled_num} instances as labeled data, {unlabeled_num} instances as unlabeled data')
        else:
            data = self.read_and_process_unlabel_set(root, filename)

            self.unlabel_data = data
            self.training_data = []
            print(f'We get {len(self.unlabel_data)} sentences.')

    def read_and_process_unlabel_set(self, root, filename):

        path = os.path.join(root, filename)

        data = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line)['sentence'])

        processed_data = self.preprocess(data, mode='serving')

        data = []
        for line in processed_data:
            sentence, spans = line
            data_point = DataPoint(
                        sentence=sentence,
                        spans=spans,
                        ner_labels=[-1]*len(spans),
                        parsed_tokens=self.parser.parse(' '.join(sentence)),
                        label_num = self.label_id_mapper.get_num()
                    )
            data_point.unlabel_reset()
            data.append(data_point)

        return data

    def preprocess(self, data, mode='training'):
        """ Preprocess the data and convert to ids. """
        processed = []
        if mode == 'training':
            for line in data:
                for sentence, ner in zip(line["sentences"], line["ner"]):

                    ner_dict = format_label_fields(ner)
                    sentence, spans, ner_labels, ner_tags = self.text_to_instance(sentence, ner_dict)
                    processed.append([sentence, spans, ner_labels, ner_tags])

        else:  #serving
            for sentence in data:
                sentence, spans = self.text_to_instance(sentence, None, mode=mode)

                processed.append([sentence, spans])
                
        return processed

    def text_to_instance(self,
                        sentence,
                        ner_dict,
                        mode='training'
                        ):
        spans = []
        ner_labels = []
        ner_tags = []
        if mode == 'training':
            for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
                span_ix = (start, end)
                spans.append((start, end))
                ner_label = ner_dict[span_ix]
                ner_labels.append(self.label_id_mapper.get_id(ner_label))
                ner_tags.append(ner_label)
            
            return sentence, spans, ner_labels, ner_tags

        else:
            for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
                span_ix = (start, end)
                spans.append((start, end))

            return sentence, spans
            
    def enumerate_spans(self, sentence, max_span_width, min_span_width=1):

        max_span_width = max_span_width or len(sentence)
        spans = []

        for start_index in range(len(sentence)):
            last_end_index = min(start_index + max_span_width, len(sentence))
            first_end_index = min(start_index + min_span_width - 1, len(sentence))
            for end_index in range(first_end_index, last_end_index):
                start = start_index
                end = end_index
                spans.append((start, end))
        return spans


    def __len__(self):
        return 100000000

    def __getitem__(self, index):

        index = random.randint(0, len(self.training_data)-1)

        raw_data = self.training_data[index]
        data = raw_data.deepcopy_all_data()
        
        tokens, idx_dict = self.encoder.tokenize(data['sentence'])
        
        converted_spans = []
        for span in data['spans']:
            converted_spans.append(self.convert_span(span, idx_dict))

        data['sentence'] = tokens
        data['spans'] = converted_spans
             
        return [raw_data, data]


    def get_unlabel_item(self, index):

        raw_data = self.unlabel_data[index]
        data = raw_data.deepcopy_all_data()

        tokens, idx_dict = self.encoder.tokenize(data['sentence'])
        
        converted_spans = []
        for span in data['spans']:
            converted_spans.append(self.convert_span(span, idx_dict))

        data['sentence'] = tokens
        data['spans'] = converted_spans
             
        return [raw_data, data]

    
    def convert_span(self, span, idx_dict):

        start_idx = span[0]
        end_idx = span[1]
        
        span_idx = idx_dict[start_idx] + idx_dict[end_idx]

        if len(span_idx)==0:  ## some special character in Bert tokenizer will become None, like white space in unicode
            return (0, 0)

        return (min(span_idx), max(span_idx))

    def collate_fn(self, data):
        
        raw_data_b, data_b = zip(*data)
        data_b = list_to_dict(data_b)
        
        max_length = max([len(tokens) for tokens in data_b['sentence']])
        ##padding
        for tokens in data_b['sentence']:
            while len(tokens)<max_length:
                tokens.append(0)
        
        data_b['sentence'] = torch.LongTensor(data_b['sentence'])
        ##mask
        data_b['mask'] = data_b['sentence'].eq(0).eq(0).float()


        span_max_length = max([len(converted_spans) for converted_spans in data_b['spans']]) or 1 ## at least length is 1
        ##span padding
        for converted_spans in data_b['spans']:
            while len(converted_spans)<span_max_length:
                converted_spans.append((0, 0))
        
        data_b['spans'] = torch.LongTensor(data_b['spans'])
        ## span label padding
        for ner_labels in data_b['ner_labels']:
            while len(ner_labels)<span_max_length:
                ner_labels.append(0)
                
        data_b['ner_labels'] = torch.LongTensor(data_b['ner_labels'])

        for soft_labels in data_b['soft_labels']:
            while len(soft_labels)<span_max_length:
                soft_labels.append([0]*self.label_id_mapper.get_num())
        
        data_b['soft_labels'] = torch.FloatTensor(data_b['soft_labels'])
        
        ##span mask
        
        for span_mask in data_b['span_mask']:
            while len(span_mask)<span_max_length:
                span_mask.append(0)

        data_b['span_mask'] = torch.FloatTensor(data_b['span_mask'])

        for span_mask_for_loss in data_b['span_mask_for_loss']:
            while len(span_mask_for_loss)<span_max_length:
                span_mask_for_loss.append(0)
        
        data_b['span_mask_for_loss'] = torch.FloatTensor(data_b['span_mask_for_loss'])

        return raw_data_b, data_b

    def update_dataset(self, new_data):

        self.training_data = deepcopy(self.label_data) + new_data

class FewShotSample:
    def __init__(self, sentence, spans, tags, data_idx):
        self.words, self.tags = sentence, ['O' for _ in range(len(sentence))]
        self.spans = spans
        for i, span in enumerate(spans):
            start, end = span
            for j in range(start, end+1):
                self.tags[j] = tags[i]
        self.data_idx = [data_idx for _ in range(len(sentence))]
        self.words = [word.lower() for word in self.words]
        self.class_count = {}

    def __count_entities__(self):
        current_tag = self.tags[0]
        for tag in self.tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I' 
        tag_class = list(set(self.tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    # guarantee that the label of this sample belongs to the class we want to sample from
    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])



class FewShotDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, data, tokenizer, label_id_mapper, N, K, Q, max_length=60, ignore_label_id=-1):
        self.N = N
        self.K = K
        self.Q = Q
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.label_id_mapper = label_id_mapper
        if isinstance(data, set):
            self.samples = self.__construct_fewshot_samples_from_instances__(data)
        else:
            self.samples = self.__construct_fewshot_samples_from_sentences__(data)
        self.sampler = FewshotSampler(N, K, Q, label_id_mapper, self.samples)  # init sampler
        
        


    def __construct_fewshot_samples_from_instances__(self, instance_set):

        samples = []
        data_id_instance_dict = defaultdict(set)

        for instance in instance_set:
            data_id_instance_dict[instance.data_idx].add(instance)

        for data_idx, instances in data_id_instance_dict.items():
            spans = []
            ground_truth_tags = []
            for instance in instances:
                spans.append(instance.span)
                ground_truth_tags.append(instance.ground_truth_tag)
            fewshot_sample = FewShotSample(instance.sentence, spans, ground_truth_tags, data_idx)
            samples.append(fewshot_sample)

        return samples

    def __construct_fewshot_samples_from_sentences__(self, dataset):
        samples = []

        for data_idx, data in enumerate(dataset):
            ground_truth_tags = []
            spans = []
            for i, ground_truth in enumerate(data.ground_truth):
                if ground_truth != self.label_id_mapper.get_neg_id():
                    ground_truth_tags.append(self.label_id_mapper.get_label(int(ground_truth)))
                    spans.append(data.spans[i])
            fewshot_sample = FewShotSample(data.sentence, spans, ground_truth_tags, data_idx)
            samples.append(fewshot_sample)

        return samples


    def __getraw__(self, sample):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags
        tokens = []
        labels = []
        for word, tag in zip(sample.words, sample.tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label.get(tag, 0)] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
                #assert len(tokens) == len(labels), print(word_tokens, word_labels)

        
        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        labels_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length-2])
            tokens = tokens[self.max_length-2:]
            labels_list.append(labels[:self.max_length-2])
            labels = labels[self.max_length-2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        span_mask_list = []
        span_label_list = []
        for i, tokens in enumerate(tokens_list):
            # token -> ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
            # padding with 0
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask, mask off the paddings
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, mask off [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1
            text_mask_list.append(text_mask)

            # span mask, mask off non-span tokens
            # if len(tokens) == self.max_length:
            #     span_mask = np.zeros((self.max_length), dtype=np.int32)
            # else:
            #     span_mask = np.zeros((len(tokens)), dtype=np.int32)
            span_mask = np.zeros((self.max_length), dtype=np.int32)
            span_label = []
            labels = labels_list[i]
            start, end = 0, 0
            for idx, label in enumerate(labels):
                if label != 0 and label != self.ignore_label_id:
                    start = idx
                    end = start + 1
                    while end < len(labels) and labels[end] == self.ignore_label_id:
                        end += 1
                    break
            span_mask[start:end] = 1
            span_mask_list.append(span_mask)
            for j in range(start, end):
                span_label.append(labels[j])
            span_label_list.append(span_label)


            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, mask_list, text_mask_list, span_mask_list, labels_list, span_label_list

    def __additem__(self, index, d, word, mask, text_mask, span_mask, label, span_label):
        d['index'].append(index)
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['span_label'] += span_label
        d['text_mask'] += text_mask
        d['span_mask'] += span_mask

    def __populate__(self, idx_list, samples, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'index':[], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'span_mask':[], 'span_label':[] }
        for idx in idx_list:
            word, mask, text_mask, span_mask, label, span_label = self.__getraw__(samples[idx])  # convert the sample
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            span_mask = torch.tensor(span_mask).long()
            self.__additem__(idx, dataset, word, mask, text_mask, span_mask, label, span_label)  # the idx is the idx in the whole sample set
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    # get a single data in a batch, in this implementation, a single data includes a support set and a query set
    def __getitem__(self, index):
        # target classes are the class names
        target_classes, support_idx, query_idx = self.sampler.__next__()  # using K~2K shot sampling method in the pape
        
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}  # class name: index, only N classes
        self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}  # index: class name, only N classes
        support_set = self.__populate__(support_idx, self.samples)  # store the sample objects to a dict
        query_set = self.__populate__(query_idx, self.samples, savelabeldic=True)  # store the sample objects to a dict
        return support_set, query_set
    
    def __len__(self):
        return 1000000000



    #  merges a list of samples to a batch, in the form of dictionary
    #  some of the fields are in the form of tensor, some are not
    def collate_fn(self, batch):  # these data are from the sampled

        batch_support = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'span_mask':[], 'span_label':[]}
        batch_query = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'label2tag':[], 'text_mask':[], 'span_mask':[], 'span_label':[]}
        support_sets, query_sets = zip(*batch)

        for i in range(len(support_sets)):
            for k in batch_support:
                batch_support[k] += support_sets[i][k]
            for k in batch_query:
                batch_query[k] += query_sets[i][k]
        for k in batch_support:
            if k not in ['label', 'span_label', 'sentence_num']:
                batch_support[k] = torch.stack(batch_support[k], 0)
        for k in batch_query:
            if k not in ['label', 'span_label', 'sentence_num', 'label2tag']:
                batch_query[k] = torch.stack(batch_query[k], 0)
        batch_support['label'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label']]
        batch_query['label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label']]
        batch_support['span_label'] = [torch.tensor(tag_list).long() for tag_list in batch_support['span_label']]
        batch_query['span_label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['span_label']]
        return batch_support, batch_query



class FewShotPredDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, data, tokenizer, label_id_mapper, N=None, K=None, Q=None, max_length=60, ignore_label_id=-1):
        self.N = N
        self.K = K
        self.Q = Q
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_id_mapper = label_id_mapper
        if isinstance(data, set):
            self.samples, self.classes = self.__construct_fewshot_samples_from_instances__(data)
            self.classes = ['0'] + self.classes
            self.tag2label = {tag:idx for idx, tag in enumerate(self.classes)}
            self.label2tag = {idx:tag for idx, tag in enumerate(self.classes)}
        else:
            self.samples = self.__construct_fewshot_samples_from_sentences__(data)
            self.classes, self.tag2label, self.label2tag = [], {}, {}
        self.ignore_label_id = ignore_label_id


    def __construct_fewshot_samples_from_instances__(self, instance_set):
        samples = []
        classes = set()
        data_id_instance_dict = defaultdict(set)

        for instance in instance_set:
            data_id_instance_dict[instance.data_idx].add(instance)

        for data_idx, instances in data_id_instance_dict.items():
            spans = []
            ground_truth_tags = []
            for instance in instances:
                spans.append(instance.span)
                tag = 'O' if instance.ground_truth_tag == '' else instance.ground_truth_tag
                ground_truth_tags.append(tag)
                classes.add(instance.ground_truth_tag)
            fewshot_sample = FewShotSample(instance.sentence, spans, ground_truth_tags, data_idx)
            samples.append(fewshot_sample)

        if '' in classes:
            classes.remove('')
        return samples, list(classes)

    def __construct_fewshot_samples_from_sentences__(self, dataset):
        samples = []

        for data_idx, data in enumerate(dataset):
            ground_truth_tags = []
            spans = []
            for i, ground_truth in enumerate(data.ground_truth):
                if ground_truth != self.label_id_mapper.get_neg_id():
                    ground_truth_tags.append(self.label_id_mapper.get_label(int(ground_truth)))
                    spans.append(data.spans[i])
            fewshot_sample = FewShotSample(data.sentence, spans, ground_truth_tags, data_idx)
            samples.append(fewshot_sample)

        return samples


    def __getraw__(self, sample):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags
        tokens = []
        labels = []
        data_ids = []
        for word, tag, data_idx in zip(sample.words, sample.tags, sample.data_idx):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label.get(tag, 0)] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
                #assert len(tokens) == len(labels), print(word_tokens, word_labels)
                ids = [data_idx] * len(word_tokens)
                data_ids.extend(ids)

        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        labels_list = []
        data_ids_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length-2])
            tokens = tokens[self.max_length-2:]
            labels_list.append(labels[:self.max_length-2])
            labels = labels[self.max_length-2:]
            data_ids_list.append(data_ids[:self.max_length-2])
            data_ids = data_ids[self.max_length-2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)
            data_ids_list.append(data_ids)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        for i, tokens in enumerate(tokens_list):
            # token -> ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1
            text_mask_list.append(text_mask)

            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, mask_list, text_mask_list, labels_list, data_ids_list

    def __additem__(self, index, d, word, mask, text_mask, label, data_idx):
        d['index'].append(index)
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['text_mask'] += text_mask
        d['data_idx'] += data_idx

    def __populate__(self, idx_list, samples):
        dataset = {'index':[], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'data_idx':[] }
        for idx in idx_list:
            word, mask, text_mask, label, data_idx = self.__getraw__(samples[idx])  # convert the sample
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            self.__additem__(idx, dataset, word, mask, text_mask, label, data_idx)  # the idx is the idx in the whole sample set
        dataset['sentence_num'] = [len(dataset['word'])]
        dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, index):
        sample_dict = self.__populate__([index], self.samples)  # store the sample objects to a dict
        return sample_dict
    
    def __len__(self):
        return len(self.samples)


    #  merges a list of samples to a batch, in the form of dictionary
    #  some of the fields are in the form of tensor, some are not
    def collate_fn(self, data):  # these data are from the sampled
        batch = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'label2tag':[], 'text_mask':[], 'data_idx':[]}

        for i in range(len(data)):
            for k in batch:
                batch[k] += data[i][k]
        for k in batch:
            if k not in ['label', 'sentence_num', 'label2tag', 'data_idx']:
                batch[k] = torch.stack(batch[k], 0)
        batch['label'] = [torch.tensor(tag_list).long() for tag_list in batch['label']]
        batch['data_idx'] = [torch.tensor(id_list).long() for id_list in batch['data_idx']]
        return batch


class MyDataLoader: ## For unlabeled data
    def __init__(self, dataset, batch_size):

        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.max_batch_i = len(dataset.unlabel_data)//batch_size
        self.max_one_i = len(dataset.unlabel_data)
    
    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.max_one_i
    
    def __next__(self):
        
        if len(self.dataset.unlabel_data)==0:  ## empty dataset check
            raise StopIteration

        batch_data = []
        if self.index < self.max_batch_i:
            for i in range(self.index * self.batch_size, (self.index+1) * self.batch_size):
                batch_data.append(self.dataset.get_unlabel_item(i))
            
        elif self.index == self.max_batch_i:
            for i in range(self.index * self.batch_size, self.max_one_i):
                batch_data.append(self.dataset.get_unlabel_item(i))
        else:
            raise StopIteration
        self.index+=1
        return self.dataset.collate_fn(batch_data)
    
    def reset(self):
        self.index = 0

    def has_next(self):
        
        return self.index * self.batch_size < self.max_one_i


def get_loader(root, filename, encoder, batch_size, label_id_mapper, is_train, opt, mode='training'):

    dataset = DataSet(root, filename, encoder, batch_size, label_id_mapper, is_train=is_train, opt=opt, mode=mode)

    label_data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=dataset.collate_fn)

    unlabel_data_loader = MyDataLoader(dataset=dataset, batch_size=batch_size)

    if is_train:
        return iter(label_data_loader), iter(unlabel_data_loader), dataset
    else:
        return iter(label_data_loader), dataset

def update_train_loader(dataset, batch_size):

    label_data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=dataset.collate_fn)

    unlabel_data_loader = MyDataLoader(dataset=dataset, batch_size=batch_size)

    return iter(label_data_loader), iter(unlabel_data_loader)



def get_fewshot_loader(fewshot_dataset, batch_size=16):

    data_loader = data.DataLoader(dataset=fewshot_dataset,  # implemented torch.data.Dataset
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=fewshot_dataset.collate_fn)  #  merges a list of samples to form a mini-batch

    return iter(data_loader)  # make it iterable