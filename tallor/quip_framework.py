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

import os
import random
import numpy as np
from .data_loader import update_train_loader
import torch
import torch.utils.data as data
from transformers import AdamW, get_linear_schedule_with_warmup
from copy import deepcopy
from collections import defaultdict
from tallor.rule_kits.rule_labeler import RuleLabeler
from tallor.fewshot_kits.quip_labeler import QuipLabeler
from tallor.utils import save_dataset, load_dataset
from tqdm import tqdm

class IEFramework:

    def __init__(self, label_id_mapper, dataset_name, opt, logger, batch_size, fewshot_batch_size,
                train_data_loader, val_data_loader, test_data_loader, unlabel_data_loader,
                mode, max_length=60, fewshot_N=20, exp_dir='.'):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.opt = opt
        self.N = fewshot_N
        self.mode = mode
        self.exp_dir = exp_dir
        self.start_step = 0
        self.label_id_mapper = label_id_mapper
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.unlabel_data_loader = unlabel_data_loader
        self.training_set = self.unlabel_data_loader.dataset
        self.unlabel_data = self.training_set.unlabel_data
        self.batch_size = batch_size
        self.rule_labeler = RuleLabeler(label_id_mapper, self.unlabel_data, dataset_name, opt, exp_dir=exp_dir)
        self.high_precision_instances, self.negative_instance_set, self.to_label_instance_set = self.rule_labeler.get_three_sets()
        self.logger = logger
        self.fewshot_labeler = QuipLabeler(label_id_mapper, self.training_set.unlabel_data, val_data_loader, test_data_loader,
                                            batch_size=fewshot_batch_size, max_length=max_length, mode=mode, exp_dir=exp_dir)

        
        if opt.load_dataset_path is not None:
            self.high_precision_instances, self.negative_instance_set, self.to_label_instance_set, _ = load_dataset(opt.load_dataset_path, 'JointIE')
            self.fewshot_labeler.high_precision_instances, self.fewshot_labeler.negative_instance_set, \
                self.fewshot_labeler.to_label_instance_set, _ = load_dataset(opt.load_dataset_path, 'QuIP')
            self.rule_labeler.high_precision_instances, self.rule_labeler.negative_instance_set, \
                self.rule_labeler.to_label_instance_set, step = load_dataset(opt.load_dataset_path, 'RuleLabeler')
            self.start_step = step + 1

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

    def train(self,
              model=None,
              model_name=None,
              fewshot_model=None,
              fewshot_model_name=None,
              val_iter=1000,
              val_step=2000,
              num_steps=32,
              load_ckpt=None,
              save_ckpt=None,
              load_fewshot_ckpt=None,
              fewshot_save_ckpt=None,
              warmup_step=100,
              update_threshold=0.7
              ):

        # Init
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
        
        if load_fewshot_ckpt:
            state_dict = self.__load_model__(load_fewshot_ckpt)['state_dict']
            own_state = fewshot_model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
     
        # Training
        best_ner_f1 = 0
        self.logger.info('Start framework training!')
        
        if self.mode == 'coaug':
            self.save_initial_model(model, save_ckpt)
            self.save_initial_model(fewshot_model, fewshot_save_ckpt)

            # Define the high precision sets that can be used by the instance augmenter models
            self.fewshot_labeler.set_three_sets(self.high_precision_instances, self.negative_instance_set, self.to_label_instance_set)

            pos_instance_set = set()
            for i in range(self.start_step, num_steps):

                # Label the data using the rule augmenter model
                self.rule_labeler.update_three_set(pos_instance_set, i)
                pos_instance_set, labeled_data, all_data = self.rule_labeler.pipeline(i)
                self.rule_labeler.eval(i, all_data, labeled_data)

                # Update the training set of the NER model and train the JointIE model
                self.update_dataset_and_loader(self.training_set, labeled_data)
                self.load_initial_model(model, save_ckpt)
                train_step = val_step + 50*i
                _, train_p, _ = self.train_ner_model(model, train_step, warmup_step)
                ner_f1, _, _ = self.eval(model, val_iter, i)
                if ner_f1 > best_ner_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    torch.save({'state_dict': fewshot_model.state_dict()}, fewshot_save_ckpt)
                    best_ner_f1 = ner_f1
                if self.opt.dataset == 'chemdner': update_threshold = train_p
                
                # Select the high precision data instances from the current set using the JointIE model
                # Also, get new rules based on the high-precision data
                self.select_and_update_training(model, update_threshold)
                model.metric_reset()

                train_step = self.opt.quip_train + 100*i 
                
                self.fewshot_labeler.update_three_set(pos_instance_set)
                self.load_initial_model(fewshot_model, fewshot_save_ckpt)
                pos_instance_set = self.fewshot_labeler.pipeline(fewshot_model, fewshot_model_name, step=i, rule_labeled_set=set(self.rule_labeler.selected_list), train_iter=train_step)

                save_dataset((self.high_precision_instances, self.negative_instance_set, self.to_label_instance_set), self.exp_dir, f'JointIE_{i}')
                save_dataset((self.rule_labeler.high_precision_instances, self.rule_labeler.negative_instance_set, self.rule_labeler.to_label_instance_set), self.exp_dir, f'RuleLabeler_{i}')
                save_dataset((self.fewshot_labeler.high_precision_instances, self.fewshot_labeler.negative_instance_set, self.fewshot_labeler.to_label_instance_set), self.exp_dir, f'QuIP_{i}')

            model.metric_reset()
        
        
        elif self.mode == 'quip':

            self.save_initial_model(fewshot_model, fewshot_save_ckpt)
            self.sample_initial_fewshots()
            self.fewshot_labeler.set_three_sets(self.high_precision_instances, self.negative_instance_set, self.to_label_instance_set)

            pos_instance_set = set()
            for i in range(self.start_step, num_steps):

                train_step = 800 + 100*i
                # Always starting the model from scratch
                self.load_initial_model(fewshot_model, fewshot_save_ckpt)
                # Set the possible training set for the pipeline
                self.fewshot_labeler.update_three_set(pos_instance_set)
                # Train the model and label the unlabeled data. Thresholding also done within.
                pos_instance_set = self.fewshot_labeler.pipeline(fewshot_model, fewshot_model_name, step=i, train_iter=train_step)
                save_dataset((self.fewshot_labeler.high_precision_instances, self.fewshot_labeler.negative_instance_set, self.fewshot_labeler.to_label_instance_set), self.exp_dir, f'QuIP_{i}')

                ner_f1, _, _ = self.fewshot_labeler.eval(fewshot_model, val_iter)
                if ner_f1 > best_ner_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': fewshot_model.state_dict()}, fewshot_save_ckpt)
                    best_ner_f1 = ner_f1

        print("\n####################")
        self.logger.info("Finish training " + model_name)
    
    def filter_step(self, model, pos_instance_set, step):
        pos_instance_dataset = list(map(self.fewshot_labeler.tokenize_and_align_labels, pos_instance_set))
        pos_instance_dataset = data.DataLoader(dataset=pos_instance_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=8,
                                                collate_fn=self.fewshot_labeler.collate_fn)
        pos_data_loader = iter(pos_instance_dataset)
        
        model.eval()

        drop_instances = set()
        with torch.no_grad():
            instances = True
            while instances:
                instances = next(pos_data_loader, None)
                if instances is None: break
                instances = {key:torch.tensor(value).long().cuda() for key, value in instances.items()}
                label = instances['labels']
                logits, pred = model(instances)
                probs = logits[label!=-100].softmax(dim=-1)
                conf_score, pred = torch.max(probs, dim=-1)
                dummy_idx = np.array(range(label.shape[0]))[:,None].repeat(label.shape[-1], axis=-1)
                dummy_idx = dummy_idx[label.cpu().numpy()!=-100]
                if (pred != label[label!=-100]).all():
                    # Now the predictions differ, discard the elements
                    drop_idxs = torch.where(pred!=label[label!=-100])[0].cpu().numpy().tolist()
                else:
                    # Now consider the confidence for the thresholding
                    threshold = min(self.opt.constraint_threshold, 0.8 + step * 0.01)
                    # threshold = self.opt.constraint_threshold
                    drop_idxs = torch.where(conf_score < max(0, threshold))[0].cpu().numpy().tolist()
                drop_idxs = [instances['data_idx'][dummy_idx[idx]].cpu().numpy().tolist() for idx in drop_idxs]
                drop_instances.update(drop_idxs)
        
        pos_instance_set = set([instance for instance in pos_instance_set if instance.data_idx not in list(drop_instances)])
        return pos_instance_set
    
    def distribute_three_sets(self, filtered_set):

        high_precision_instances = defaultdict(set)
        for k, instances in self.high_precision_instances.items():
            for instance in instances:
                if instance not in filtered_set:
                    self.to_label_instance_set.add(instance)
                else:
                    high_precision_instances[k].add(instance)
        
        self.high_precision_instances = high_precision_instances
            
    def train_ner_model(self, model, train_iter, warmup_step):

        model.train()

        parameters_to_optimize = []
        for n, p in list(model.named_parameters()):
            if p.requires_grad:
                parameters_to_optimize.append((n, p))
        

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        for _ in tqdm(range(train_iter), ncols=100, total=train_iter, desc='Train NER model.'):
            
            _, data_b = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    if k != 'ner_tags':
                        data_b[k] = v.cuda()
            
            output_dict  = model(data_b['sentence'], data_b['mask'], data_b['spans'], data_b['span_mask'], 
                                data_b['span_mask_for_loss'], data_b['ner_labels'], data_b['soft_labels'])
            
            loss = output_dict['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            ner_results = output_dict['span_metrics']
           
            ner_prf = ner_results[1].get_metric()
        
        return ner_prf['f'], ner_prf['p'], ner_prf['r']

    def eval(self,
            model,
            eval_iter,
            step,
            ckpt=None): 

        model.metric_reset()
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        with torch.no_grad():
            for it in tqdm(range(eval_iter), ncols=100, total=eval_iter, desc='Evaluation.'):
                
                _, data_b = next(eval_dataset)
        
                if torch.cuda.is_available():
                    for k, v in data_b.items():
                        if k != 'ner_tags':
                            data_b[k] = v.cuda()

                output_dict  = model(data_b['sentence'], data_b['mask'], data_b['spans'], data_b['span_mask'], 
                                data_b['span_mask_for_loss'], data_b['ner_labels'], data_b['soft_labels'])

                # loss = output_dict['loss']

                ner_results = output_dict['span_metrics']
                
                ner_prf = ner_results[1].get_metric()
                ner_prf_b = ner_results[2].get_metric()
                
        self.logger.info('STEP:{0}. Current performance on NER: F1: {1:2.4f}, P: {2:2.4f}, R: {3:2.4f}.'.format(step, ner_prf['f'], ner_prf['p'], ner_prf['r']))
        self.logger.info('Binary result: F1: {0:2.4f}, P: {1:2.4f}, R: {2:2.4f}.'.format(ner_prf_b['f'], ner_prf_b['p'], ner_prf_b['r']))
        return ner_prf['f'], ner_prf['p'], ner_prf['r']


    def select_and_update_training(self, model, update_threshold):

        model.eval()         
        new_data_self_training, _ = self.self_training(model, update_threshold)
        
        self.rule_labeler.update_rule_pipeline(new_data_self_training)


    def update_three_sets(self, rule_high_precision_instances={}, rule_negative_instance_set=set(), rule_to_label_instance_set=set(),
                                fewshot_high_precision_instances={}, fewshot_negative_instance_set=set(), fewshot_to_label_instance_set=set()):

        del self.high_precision_instances
        del self.negative_instance_set
        del self.to_label_instance_set

        self.high_precision_instances = defaultdict(set)
        self.negative_instance_set = set()
        self.to_label_instance_set = set()

        for k, rule_instances in rule_high_precision_instances.items():
            for instance in rule_instances:
                self.high_precision_instances[k].add(instance)

        for k, fewshot_instances in fewshot_high_precision_instances.items():
            for instance in fewshot_instances:
                self.high_precision_instances[k].add(instance)
        
        if rule_negative_instance_set and fewshot_negative_instance_set:

            rule_high_precision_instance_set = set()
            for _, v in rule_high_precision_instances.items():
                for instance in v:
                    rule_high_precision_instance_set.add(instance)

            high_precision_instance_set = set()
            for _, v in self.high_precision_instances.items():
                for instance in v:
                    high_precision_instance_set.add(instance)
                    if instance in self.rule_labeler.candidate_set:
                        self.rule_labeler.candidate_set.remove(instance)
                    # if instance.label in self.fewshot_labeler.candidate_instances and instance in self.fewshot_labeler.candidate_instances[instance.label]:
                        # self.fewshot_labeler.candidate_instances[instance.label].remove(instance)

            self.negative_instance_set = rule_negative_instance_set & fewshot_negative_instance_set
            labeled_set = high_precision_instance_set | self.negative_instance_set
            for instance in (rule_high_precision_instance_set | rule_negative_instance_set | rule_to_label_instance_set):
                if instance not in labeled_set:
                    self.to_label_instance_set.add(instance)

            print(f'After rule and QuIP labeling, we get {len(high_precision_instance_set)} positive instances, {len(self.negative_instance_set)} negative instances, {len(self.to_label_instance_set)} unlabel instances.')


        elif rule_negative_instance_set:

            for instance in rule_negative_instance_set:
                self.negative_instance_set.add(instance)

            for instance in rule_to_label_instance_set:
                self.to_label_instance_set.add(instance)

        elif fewshot_negative_instance_set:

            for instance in fewshot_negative_instance_set:
                self.negative_instance_set.add(instance)

            for instance in fewshot_to_label_instance_set:
                self.to_label_instance_set.add(instance)

        else:

            raise ValueError('Empty high precision sets!')


    def update_dataset_and_loader(self, dataset, new_data):
    
        dataset.update_dataset(new_data)

        del self.train_data_loader

        self.train_data_loader, self.unlabel_data_loader = update_train_loader(dataset, self.batch_size)

        self.logger.info(f'NER training data updated (instances with spans merged)! {len(dataset.training_data)} data points for training, {len(new_data)} data points are newly labeled.')


    def self_training(self, model, update_threshold):

        raw_data = []
        ner_res = []
        print('Begin predict all data.')
        while self.unlabel_data_loader.has_next():
            
            raw_data_b, data_b = next(self.unlabel_data_loader)
            
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    if k != 'ner_tags':
                        data_b[k] = v.cuda()
            
            output_dict  = model.predict(data_b['sentence'], data_b['mask'], data_b['spans'], data_b['span_mask'])
            
            ner_res_list = model.decode(output_dict)  ## [[{'span_idx': 0, 'span': (1, 4), 'prob': 0.9999961853027344, 'class': 1} * instance_num] * 200 sentences]
            
            raw_data += raw_data_b
            ner_res += ner_res_list
        print('Done.')
        new_data = self.select_and_update_data(raw_data, ner_res, update_threshold)

        return new_data, raw_data

    def select_and_update_data(self, raw_data, ner_res, update_threshold):

        ner_res_dict = defaultdict(list) # key is the class, value is the res instances

        data_id_ner_dict = defaultdict(list)  # key is the data_id, value is the res intances (for update relation)
    
        for i, ner_res_entry in enumerate(ner_res):  ## [[{'span_idx': 0, 'span': (1, 4), 'prob': 0.9999961853027344, 'class': 1} * instance_num] * 200 sentences]

            for ner in ner_res_entry:
                ner['data_id'] = i
                ner_res_dict[ner['class']].append(ner)
        
        for value in ner_res_dict.values():
            value = sorted(value, key=lambda x: x['prob'], reverse=True)
            selected_ner_res = value[:int(len(value)*update_threshold)]
            for instance in selected_ner_res:
                data_id_ner_dict[instance['data_id']].append(instance)

        new_data = []
        new_raw_data = deepcopy(raw_data)
        for i, data_entry in enumerate(new_raw_data):

            if i in data_id_ner_dict:
                new_data_entry = self.update_data_entry(data_entry, data_id_ner_dict[i])
            else:
                new_data_entry = data_entry

            new_data.append(new_data_entry)
        return new_data

    
    def update_data_entry(self, data_entry, ner_res_list):
    
        if len(ner_res_list)>0:

            for ner_res in ner_res_list:
                span_idx = ner_res['span_idx']
                label = ner_res['class']
                data_entry.ner_labels[span_idx] = label
                data_entry.span_mask_for_loss[span_idx] = 1

        return data_entry

    def save_initial_model(self, model, path):

        torch.save({'state_dict': model.state_dict()}, path+'_initial')

    def load_initial_model(self, model, path):

        state_dict = self.__load_model__(path+'_initial')['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    def sample_initial_fewshots(self):

        N = self.N
        full_data = list(self.negative_instance_set | self.to_label_instance_set)
        class_list = set(range(self.label_id_mapper.get_num()))
        class_list.remove(self.label_id_mapper.get_neg_id())
        
        sampled_labels = random.sample(class_list, min(self.label_id_mapper.get_num()-1, N))
        count_dict = { k:0 for k in sampled_labels}

        positive_data = []
        for instance in full_data:
            if instance.ground_truth != self.label_id_mapper.get_neg_id():
                positive_data.append(instance)

        stop_sampling = lambda count_dict: all([True if v >= N//len(count_dict.keys()) else False for _, v in count_dict.items()])
        
        while not stop_sampling(count_dict):
            instance = random.choice(positive_data)
            label_id = instance.ground_truth
            if label_id not in count_dict:
                continue
            elif count_dict[label_id] == N//len(count_dict.keys()):
                continue
            else:
                count_dict[label_id] += 1
                instance.label = label_id
                instance.label_prob = self.label_id_mapper.get_soft_label(int(label_id))
                self.high_precision_instances[label_id].add(instance)

        for _, v in self.high_precision_instances.items():
            for instance in v:
                if instance in self.negative_instance_set:
                    self.negative_instance_set.remove(instance)
                if instance in self.to_label_instance_set:
                    self.to_label_instance_set.remove(instance)

    def label_instance_to_labeled_data(self, data):
        '''
        Output: A list of labeled DataPoint
        '''
        data_with_positive_idx = set()
        data_with_positive = []

        positive_instance_set = set()
        for _, instance_set in self.high_precision_instances.items():
            positive_instance_set |= instance_set

        for instance in positive_instance_set:
                
            data[instance.data_idx].ner_labels[instance.span_idx] = instance.label
            data[instance.data_idx].soft_labels[instance.span_idx] = deepcopy(instance.label_prob)
            data[instance.data_idx].span_mask_for_loss[instance.span_idx] = 1
            data_with_positive_idx.add(instance.data_idx)

        for instance in self.negative_instance_set:

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


            



        
        








