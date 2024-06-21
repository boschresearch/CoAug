#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#

from collections import defaultdict
import os
from copy import deepcopy
import numpy as np
import sys
import datetime
import logging
import torch
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from tallor.data_loader import FewShotDataset, FewShotPredDataset, get_fewshot_loader
# from .fewshot_data_loader import get_loader
np.set_printoptions(threshold=sys.maxsize)


def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0





class FewShotLabeler:

    def __init__(self, label_id_mapper, tokenizer, setting, unlabel_data, train_data_loader, val_data_loader, test_data_loader, val_dataset, batch_size, max_length=60, mode='cotrain', N=None, train_fname=None, ignore_index=-1, exp_dir='.'):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.label_id_mapper = label_id_mapper
        self.tokenizer = tokenizer
        self.trainN, self.N, self.K, self.Q = setting
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
        self.val_dataset = FewShotDataset(val_dataset.training_data, self.tokenizer, self.label_id_mapper, self.N, self.K, self.Q, self.max_length)
        self.val_data_loader = get_fewshot_loader(self.val_dataset)
        self.test_dataset = FewShotDataset(test_data_loader._dataset.training_data, self.tokenizer, self.label_id_mapper, self.N, self.K, self.Q, self.max_length)
        self.test_data_loader = get_fewshot_loader(self.test_dataset)
        self.proto_dataset = None
        self.proto_data_loader = None
        self.labeling_dataset = FewShotPredDataset(deepcopy(self.unlabel_data), self.tokenizer, self.label_id_mapper,
                                                self.N, self.K, self.Q,
                                                max_length=self.max_length)
        self.labeling_data_loader = None

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


    def reset_train_dataset_and_loader(self):

        if self.train_data_loader:
            del self.train_data_loader

        high_precision_instance_set = self.get_high_precision_set()

        # negative_examples = set(random.sample(list(self.negative_instance_set), 2 * len(high_precision_instance_set)))

        train_dataset = FewShotDataset(high_precision_instance_set, self.tokenizer, self.label_id_mapper, self.N, self.K, self.Q, self.max_length)

        self.train_data_loader = get_fewshot_loader(train_dataset)


    def reset_prototype_dataset_and_loader(self):

        if self.proto_data_loader:
            del self.proto_data_loader

        high_precision_instance_set = self.get_high_precision_set()

        # negative_examples = set(random.sample(list(self.negative_instance_set), 2 * len(high_precision_instance_set)))

        self.proto_dataset = FewShotPredDataset(high_precision_instance_set, self.tokenizer, self.label_id_mapper, self.N, self.K, self.Q, self.max_length)


        self.proto_data_loader = get_fewshot_loader(self.proto_dataset)


    def reset_labeling_dataset_and_loader(self):

        if self.labeling_data_loader:
            del self.labeling_data_loader

        self.labeling_dataset.classes = deepcopy(self.proto_dataset.classes)
        self.labeling_dataset.tag2label = deepcopy(self.proto_dataset.tag2label)
        self.labeling_dataset.label2tag = deepcopy(self.proto_dataset.label2tag)

        self.labeling_data_loader = get_fewshot_loader(self.labeling_dataset)


    def get_high_precision_set(self):
        '''
        positive set is the embeddings of high precision set
        '''
        high_precision_set = set()
        for _, instance_set in self.high_precision_instances.items():
            high_precision_set |= instance_set

        return high_precision_set


    def pipeline(self,
                 model,
                 model_name,
                 step=0,
                 rule_labeled_set=set(),
                 learning_rate=1e-4,
                 train_iter=12000,
                 val_iter=500,
                 test_iter=5000,
                 val_step=2000,
                 load_ckpt=None,
                 save_ckpt=None,
                 warmup_step=300,
                 grad_iter=1,
                 fp16=False,
                 use_sgd_for_bert=False):


        print('Begin training ProtoBERT!')
        self.reset_train_dataset_and_loader()
        self.train(model, model_name, train_iter=train_iter, warmup_step=warmup_step)
        print('ProtoBERT training done!')

        print('Begin labeling using ProtoBERT!')
        self.reset_prototype_dataset_and_loader()
        self.reset_labeling_dataset_and_loader()
        label_instances_dict, _, _, _, _, _, _, _ = self.labeling(model, model_name, rule_labeled_set)
        print('ProtoBERT labeling done!')

        self.select_instance_and_update_three_sets(label_instances_dict)
        high_precision_instance_set = self.get_high_precision_set()

        print(f'STEP:{step}: After ProtoBERT labeling, we get {len(high_precision_instance_set)} positive instances, {len(self.negative_instance_set)} negative instances, {len(self.to_label_instance_set)} unlabel instances.')

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
              learning_rate=1e-4,
              train_iter=12000,
              val_iter=500,
              val_step=1000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=1200,
              grad_iter=1,
              fp16=False,
              use_sgd_for_bert=False):
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
        print('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model = model.cuda()
        model.train()

        # Training
        best_f1 = 0.0
        iter_loss = 0.0
        iter_sample = 0
        pred_cnt = 1e-6
        label_cnt = 1e-6
        correct_cnt = 1e-10
        for it in range(start_iter, start_iter + train_iter):
            support, query = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    if k not in ['label', 'span_label', 'sentence_num', 'data_idx']:
                        support[k] = support[k].cuda()
                        query[k] = query[k].cuda()
                label = torch.cat(query['label'], 0)
                label = label.cuda()
                span_label = torch.cat(query['span_label'], 0)
                span_label = label.cuda()

            logits, pred = model(support, query)
            assert logits.shape[0] == label.shape[0], print(logits.shape, label.shape)
            loss = model.loss(logits, label) / float(grad_iter)
            tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            #iter_right += self.item(right.data)
            pred_cnt += tmp_pred_cnt
            label_cnt += tmp_label_cnt
            correct_cnt += correct
            iter_sample += 1
            if (it + 1) % 100 == 0:
                precision = correct_cnt / pred_cnt
                recall = correct_cnt / label_cnt
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
            eval_dataset = self.val_data_loader
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
                support, query = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num' and k != 'span_label':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = torch.cat(query['label'], 0)
                    label = label.cuda()
                logits, pred = model(support, query)

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, query)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

            precision = correct_cnt / pred_cnt
            recall = correct_cnt /label_cnt
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            fp_error = fp_cnt / total_token_cnt
            fn_error = fn_cnt / total_token_cnt
            within_error = within_cnt / total_span_cnt
            outer_error = outer_cnt / total_span_cnt
            sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
            sys.stdout.flush()
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


    def labeling(self,
            model,
            eval_iter,
            rule_labeled_set,
            ckpt=None):


        model.eval()

        # eval_data_loader = self.test_data_loader

        pred_cnt = 1e-6
        label_cnt = 1e-6
        correct_cnt = 1e-10

        real_tags = []
        pred_tags = []
        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct
        label_instances_dict = defaultdict(list)

        with torch.no_grad():

            it = 0

            print("Start labeling...")

            support = next(self.proto_data_loader, None)

            while support:
                model.compute_batch_prototypes(support, self.label_id_mapper.label_num-1)
                support = next(self.proto_data_loader, None)
            model.compute_prototypes()
            print('Prototypes computed!')

            unlabel_data_copy = deepcopy(self.unlabel_data)
            query = next(self.labeling_data_loader, None)
            while query:
                if torch.cuda.is_available():
                    for k in query:
                        if k not in ['label', 'sentence_num', 'label2tag', 'data_idx']:
                            query[k] = query[k].cuda()
                    label = torch.cat(query['label'], 0)
                    label = label.cuda()
                    data_idx = torch.cat(query['data_idx'], 0)
                    data_idx = data_idx.cuda()

                nearest_dist, pred = model.predict(query)
                label_instances_dict = self.update_prediction_result(label_instances_dict, unlabel_data_copy, rule_labeled_set, label, pred, nearest_dist, data_idx)

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, query)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

                query = next(self.labeling_data_loader, None)

            del unlabel_data_copy

            precision = correct_cnt / pred_cnt
            recall = correct_cnt /label_cnt
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            fp_error = fp_cnt / total_token_cnt
            fn_error = fn_cnt / total_token_cnt
            within_error = within_cnt / total_span_cnt
            outer_error = outer_cnt / total_span_cnt
            sys.stdout.write('[LABELING] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
            sys.stdout.flush()
        return label_instances_dict, precision, recall, f1, fp_error, fn_error, within_error, outer_error

    def update_prediction_result(self, label_instances_dict, unlabel_datapoints, rule_labeled_set, label, pred, nearest_dist, data_id):

        new_pred = pred[label!=self.ignore_index]
        new_data_id = data_id[label!=self.ignore_index]
        data_id_list = torch.unique(new_data_id).tolist()
        neg_id = 0

        assert new_pred.size() == new_data_id.size()

        for data_idx in data_id_list:

            spans = []
            pred_labels = []
            dist = []

            sentence_pred = new_pred[new_data_id==data_idx].tolist()
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
                            d += nearest_dist[span_right]
                            span_right += 1
                        # dist.append(d / (span_right-span_left))
                        spans.append((span_left, span_right-1))
                        pred_labels.append(sentence_pred[span_left])
                        span_left = span_right
                        span_right -= 1

                sentence_pred_with_ignored = pred[data_id==data_idx].tolist()
                new_label = label[data_id==data_idx]
                assert len(sentence_pred_with_ignored) == len(new_label), print(2, len(sentence_pred_with_ignored), len(new_label))

                span_left = 0

                while span_left < len(sentence_pred_with_ignored):
                    if sentence_pred_with_ignored[span_left] != neg_id and new_label[span_left] != self.ignore_index:
                        d = 0
                        span_right = span_left
                        while span_right < len(sentence_pred_with_ignored) and \
                            (sentence_pred_with_ignored[span_right] == sentence_pred_with_ignored[span_left] or new_label[span_right] == self.ignore_index):
                            d += nearest_dist[span_right]
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
                            pred_tag = self.labeling_dataset.label2tag[fewshot_label]
                            pred_label = self.label_id_mapper.get_id(pred_tag)
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


    def select_instance_and_update_three_sets(self, label_instances_dict):

        for label, instances_with_dist in label_instances_dict.items():
            # instances_with_dist.extend(list(self.candidate_instances[label]))
            instances_with_dist = sorted(instances_with_dist, key=lambda x: x[1], reverse=True)
            for idx, (instance, dist) in enumerate(instances_with_dist):
                if idx > min(len(instances_with_dist) * 0.2, 10):
                    if instance in self.negative_instance_set:
                        # self.candidate_instances[label].add((instance, dist))
                        self.negative_instance_set.remove(instance)
                        self.to_label_instance_set.add(instance)
                else:
                    if instance not in self.high_precision_instances[label]:
                        self.high_precision_instances[label].add(instance)
                        print('ProtoBERT labeled: ', instance.data_idx, self.label_id_mapper.get_label(instance.label), instance.ground_truth_tag, instance.span, instance.sentence[instance.span[0]:instance.span[1]+1], dist.item(), instance.sentence)

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
