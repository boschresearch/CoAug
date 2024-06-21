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

from copy import deepcopy
from tallor.label_functions.LF_manager import LFManager
from snorkel.labeling import LFApplier
from snorkel.labeling.model import MajorityLabelVoter
from tallor.rule_kits.rule_selector import RuleSelector
from tallor.rule_kits.instance_linker import InstanceLinker
from tallor.rule_kits.instance_selector import InstanceSelector
from tallor.precision_recall_f1 import DataPrecisionRecallF1
from tallor.utils import LabelInstance
from collections import defaultdict
import logging
import sys
import time
import os
        

class RuleLabeler:
    '''
    Label unlabeled data. 
    Input instance: DataPoint in utils
    '''
    def __init__(self, label_id_mapper, unlabel_data, dataset, opt, label_model_epoch=500, mode='training', exp_dir='.'):
        
        timestamp = str(time.time())
        self.result_path = f'{exp_dir}/labeler_results/{timestamp}'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.mode = mode
        self.linker = InstanceLinker(opt, dataset)
        self.logger = set_logger(exp_dir, timestamp)
        self.instance_selector = InstanceSelector(opt, timestamp, label_id_mapper, exp_dir)
        self.label_id_mapper = label_id_mapper
        self.label_model_epoch = label_model_epoch
        self.lf_manager = LFManager(label_id_mapper, dataset, mode=mode)
        self.rule_selector = RuleSelector(opt, label_id_mapper, self.logger)
        self.rule_threshold = opt.rule_threshold
        self.metric = DataPrecisionRecallF1(self.label_id_mapper.get_neg_id())

        print('Genrate all rule candidates.')
        self.unlabel_data = unlabel_data
        label_instances, self.instance_dict = self.data_to_label_instance(unlabel_data)
        self.linker.pipeline(self.instance_dict)
        instance_candidates, negative_instances = self.negative_instance_filter(label_instances)

        self.to_label_instance_set = set(instance_candidates)  ## in any time, to_label_instance_set | negative_instance_set | high_precision_instances should be the whole set
        self.negative_instance_set = set(negative_instances)
        self.high_precision_instances = defaultdict(set)

        self.rule_candidates = self.lf_manager.generate_all_new_rules(instance_candidates, self.rule_threshold)
        length = []
        for rule_candidates in self.rule_candidates:
            length.append(len(rule_candidates))
        
        print(sum(length)) 
        print('Done.')


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


    def update_three_set(self, positive_list, step):
        
        ## Dynamic rule selection
        self.selected_list, self.candidate_set = self.instance_selector.score_pipeline(self.high_precision_instances, positive_list, step)

        # high_precision_instance_set = defaultdict(set)
        for instance in self.selected_list:

            self.high_precision_instances[instance.label].add(instance)
            if instance in self.negative_instance_set:
                self.negative_instance_set.remove(instance)
            if instance in self.to_label_instance_set:
                self.to_label_instance_set.remove(instance)
        
        for instance in self.candidate_set:

            if instance in self.negative_instance_set:
                self.negative_instance_set.remove(instance)
                self.to_label_instance_set.add(instance)
        
        # return high_precision_instance_set

    def get_positive_set(self):

        positive_set = set()
        for _, instance_set in self.high_precision_instances.items():
            positive_set |= instance_set

        return positive_set

    def pipeline(self, step, rule_recorder=None):

        ## prepare data
        label_functions = self.lf_manager.get_all_functions()
        ## build label functions and models
        applier = LFApplier(lfs=label_functions)
        label_model = MajorityLabelVoter(cardinality=self.label_id_mapper.get_num())

        ## train model and labelling
        print('Begin training label model and labeling!')
        positive_list = self.labeling(list(self.to_label_instance_set | self.negative_instance_set), applier, label_model)
        print('Done.')

        self.update_three_set(positive_list, step)
        pos_instance_set = self.get_positive_set()
        self.logger.info(f'STEP:{step}: After rule labeling, we get {len(pos_instance_set)} positive instances, {len(self.negative_instance_set)} negative instances, {len(self.to_label_instance_set)} unlabel instances.')
        print(f'STEP:{step}: After rule labeling, we get {len(pos_instance_set)} positive instances, {len(self.negative_instance_set)} negative instances, {len(self.to_label_instance_set)} unlabel instances.')

        current_rules = self.lf_manager.get_all_rules()
        self.lf_manager.clear_all_rules()

        rule_log = ''
        for current_rule in current_rules:
            rule_log+=str(current_rule)+'\n'
            if self.mode != 'training' and isinstance(current_rule, dict):  ## serving
                rule_recorder.update(current_rule)
                
        self.logger.info('Current rules:\n'+rule_log)

        # Gather labeled data from rule labeling and learn JointIE model
        labeled_data, all_data = self.label_instance_to_labeled_data(deepcopy(self.unlabel_data), pos_instance_set, self.negative_instance_set)
        labeled_data = self.merge_labeled_data(labeled_data)

        return pos_instance_set, labeled_data, all_data


    def eval(self, step, all_data, labeled_data):

        num = 0
        file_path = self.result_path+f'/labeler_results{num}.txt'

        while os.path.exists(file_path):
            num+=1
            file_path = self.result_path+f'/labeler_results{num}.txt'

        f = open(file_path, 'w')
        for data_point in all_data:
            if data_point.ground_truth!=-1:
                self.metric(data_point.ner_labels, data_point.ground_truth, data_point.span_mask_for_loss)

                flag = False
                for j, label in enumerate(data_point.ner_labels):
                    span_1 = data_point.spans[j][0]
                    span_2 = data_point.spans[j][1]
                    entity = ' '.join(data_point.sentence[span_1:span_2+1])
                    soft_label = data_point.soft_labels[j]
                    ground_truth = data_point.ground_truth[j]
                    if data_point.span_mask_for_loss[j]>0 and (ground_truth != self.label_id_mapper.get_neg_id() or label != self.label_id_mapper.get_neg_id()):
                        f.write(entity+'\t'+'['+str(span_1)+' '+str(span_2)+']'+'\t'+str(soft_label)+' '+self.label_id_mapper.get_label(label)+'----'+self.label_id_mapper.get_label(ground_truth)+'\n')
                        flag = True
                
                if flag:
                    f.write(' '.join(data_point.sentence)+'\n')
                    f.write('========================\n')

        f.close()
        
        label_evalutaion = self.metric.get_metric(reset=True)
        f1 = label_evalutaion['f']
        p = label_evalutaion['p']
        r = label_evalutaion['r']
        self.logger.info(f'STEP:{step}: We get {len(labeled_data)} sentences with positive label. F1:{f1}, P: {p}, R:{r}.')
        print(f'STEP:{step}: We get {len(labeled_data)} sentences with positive label. F1:{f1}, P: {p}, R:{r}.')

    def update_rule_pipeline(self, data):

        print('Begin update rules!')

        pos_instances = []
        neg_instances = []
        instance_candidates = self.to_label_instance_set | self.negative_instance_set

        for instance in instance_candidates:

            data_idx = instance.data_idx
            span_idx = instance.span_idx
            
            data_point = data[data_idx]
            label = data_point.ner_labels[span_idx]
            
            if data_point.span_mask_for_loss[span_idx] > 0 and label!=None:

                assert instance.sentence[0] == data_point.sentence[0]

                if label != self.label_id_mapper.get_neg_id():
                
                    instance.label = label
                    instance.label_prob = self.label_id_mapper.get_soft_label(instance.label)
                    pos_instances.append(instance)

                else:
                    
                    neg_instances.append(instance)


        for instance in pos_instances:
            if instance in self.negative_instance_set:
                self.negative_instance_set.remove(instance)
                self.to_label_instance_set.add(instance)

        pos_instance_set = self.get_positive_set()
        pos_instances += list(pos_instance_set)
        
        self.logger.info(f'After NER model, we get {len(pos_instance_set)} positive instances, {len(self.negative_instance_set)} negative instances, {len(self.to_label_instance_set)} unlabel instances.')
        
        
        selected_rules = self.rule_selector.pipeline(pos_instances, neg_instances, self.rule_candidates)
        self.lf_manager.update_all_functions(selected_rules)

        ## delete selected_rules from candidates
        for i, new_rule in enumerate(selected_rules):
            for label, rules in new_rule.items():
                for rule_name in rules:
                    del self.rule_candidates[i][rule_name]

        print('Done.')


    def get_labeled_data(self):

        return self.labeled_data

    def data_to_label_instance(self, data):
        '''
        Input: A List DataPoint
        Output: A list of label instance
        '''
        instances = []
        instances_dict = defaultdict(dict)
        for data_idx, data_point in enumerate(data):
            sentence = data_point.sentence
            spans = data_point.spans
            parsed_tokens = data_point.parsed_tokens
            ner_labels = data_point.ner_labels
            ner_tags = data_point.ner_tags

            for span_idx, span in enumerate(spans):
                instance = LabelInstance(sentence, parsed_tokens, span, data_idx=data_idx, span_idx=span_idx, ground_truth=ner_labels[span_idx], ground_truth_tag=ner_tags[span_idx])
                instances.append(instance)
                instances_dict[data_idx][tuple(span)] = instance

        return instances, instances_dict

    def labeling(self, label_instances, applier, label_model):

        L_train = applier.apply(label_instances)

        labels, label_probs = label_model.predict(L_train, return_probs=True, tie_break_policy='abstain')  ## labels (instance_num, 1),  label_probs (instance_num, class_num)
        labels = list(labels.reshape(-1))
        label_probs = list(label_probs.reshape(-1, self.label_id_mapper.get_num()))
        
        positive_list = []
    
        for instance, label, prob in zip(label_instances, labels, label_probs):
            
            if label!=-1: #'ABSTAIN'
                if instance.parent==None or instance.parent==True:
                    instance.label = label
                    instance.label_prob = self.label_id_mapper.get_soft_label(int(label))
                    positive_list.append(instance)

                else:

                    parent = instance.parent
                    parent.label = label
                    parent.label_prob = self.label_id_mapper.get_soft_label(int(label))
                    positive_list.append(parent)

        return positive_list
    
    def label_instance_to_labeled_data(self, data, pos_instance_set, negative_instance_set):
        '''
        Output: A list of labeled DataPoint
        '''
        data_with_positive_idx = set()
        data_with_positive = []

        for instance in pos_instance_set:
                
            data[instance.data_idx].ner_labels[instance.span_idx] = instance.label
            data[instance.data_idx].soft_labels[instance.span_idx] = deepcopy(instance.label_prob)
            data[instance.data_idx].span_mask_for_loss[instance.span_idx] = 1
            data_with_positive_idx.add(instance.data_idx)

        for instance in negative_instance_set:

            data[instance.data_idx].ner_labels[instance.span_idx] = self.label_id_mapper.get_neg_id()
            data[instance.data_idx].soft_labels[instance.span_idx] = self.label_id_mapper.get_soft_label(self.label_id_mapper.get_neg_id())
            data[instance.data_idx].span_mask_for_loss[instance.span_idx] = 1

        for idx in data_with_positive_idx:

            data_with_positive.append(data[idx])

        
        return data_with_positive, data
    
    def merge_labeled_data(self, labeled_data):
        '''
        Merge overlapping spans.
        '''
        merged_data = []
        for data in labeled_data:
            pos_spans = []
            for i, span in enumerate(data.spans):
                if data.span_mask_for_loss[i]>0 and data.ner_labels[i] != self.label_id_mapper.get_neg_id():
                    pos_spans.append([span, data.soft_labels[i]])
    
            merged_spans = merge_spans(pos_spans)
            
            for merged_span in merged_spans:

                for j, span in enumerate(data.spans):

                    if span_equal(merged_span[0], span):  ## modify soft_labels of merged spans

                        data.soft_labels[j] = merged_span[1]

                    elif data.span_mask_for_loss[j]==0 and (span_overlap(merged_span[0], span) or span_overlap(span, merged_span[0])): ## set all spans have overlaps with positive span to negative.

                        data.span_mask_for_loss[j]=1
                        data.ner_labels[j] = self.label_id_mapper.get_neg_id()
                        data.soft_labels[j] = self.label_id_mapper.get_soft_label(data.ner_labels[j])
           
            merged_data.append(data)

        return merged_data

    def negative_instance_filter(self, label_instances):
        '''
        Only keep instances that are noun phrase or
        exist in the full dictionary.
        '''
        negative_instances = []
        to_label_instances = []

        for label_instance in label_instances:

            if label_instance.parent != None:

                span = label_instance.span
                to_label_instances.append(label_instance)

            elif self.is_noun_phrase(label_instance):

                to_label_instances.append(label_instance)

            elif self.lf_manager.pre_match(label_instance):
                span = label_instance.span
                to_label_instances.append(label_instance)

            else:

                label_instance.label = self.label_id_mapper.get_neg_id()
                # label_instance.label_prob = self.label_id_mapper.get_soft_label(label_instance.label)
                negative_instances.append(label_instance)

        return to_label_instances, negative_instances


    def is_noun_phrase(self, label_instance):

        noun_spans = []

        parsed_tokens = label_instance.parsed_tokens
        for phrase in parsed_tokens.noun_chunks:
            
            noun_spans.append([phrase.start, phrase.end-1, phrase.root])  # end exclusive to inclusive

        span = label_instance.span

        for noun_span in noun_spans:

            if noun_span[0]<=span[0] and noun_span[1]>=span[1]:
                # label_instance.entity_root = noun_span[2]
                return True

        return False


def set_logger(exp_dir, timestamp):
    ## set logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('RuleSelector')
    
    file_handler = logging.FileHandler(f'{exp_dir}/logging/RuleSelector-{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

    return logger

def merge_spans(spans):

    if len(spans)<=1:
        return spans
    
    spans = sorted(spans, key=lambda x: x[0])  ## sorted according to span idx
    
    res = []
    
    first_span = spans[0]
    tmp = [first_span]
    last = first_span[0][1]
    ## group spans that have overlaps together
    for i in range(1, len(spans)):
        span = spans[i]
        if span[0][0]>last:  ## don't have overlaps
            res.append(tmp)
            tmp = []
            tmp.append(span)
        else:
            tmp.append(span)
        if span[0][1]>last:
            last=span[0][1]
        if i==len(spans)-1:
            res.append(tmp)
    
    ## modify probabilities in soft labels (divide by number of overlapping spans)
    for spans in res:
        for span in spans:
            for i in range(len(span)):
                span[1][i] = span[1][i]/len(spans)
                
    flat_res = []
    for spans in res:
        for span in spans:
            flat_res.append(span)

    return flat_res

def span_overlap(span_1, span_2):
    '''
    if span_1 has overlap with span_2
    '''
    if (span_1[0]>=span_2[0] and span_1[0]<=span_2[1]) or (span_1[1]>=span_2[0] and span_1[1]<=span_2[1]):
        return True
    else:
        return False
        
def span_contain(span_1, span_2):

    '''
    if span_1 contains span_2
    '''
    
    if span_1[0]<=span_2[0] and span_1[1]>=span_2[1]: 
        return True
    else:
        return False


def span_equal(span_1, span_2):
    '''
    if span_1 equal span_2
    '''
    if span_1[0]==span_2[0] and span_1[1]==span_2[1]: 
        return True
    else:
        return False



    