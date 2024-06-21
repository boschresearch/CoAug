#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#

import torch
from torch import nn




class Proto(nn.Module):
    def __init__(self, my_word_encoder, proto_loss_mode, ignore_index=-1):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.ignore_index = ignore_index
        self.proto_loss_mode = proto_loss_mode
        self.word_encoder = nn.DataParallel(my_word_encoder)
        self.cost = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.proto_batch_list = []
        self.prototypes = None

    def compute_batch_prototypes(self, batch, max_label):

        batch_emb = self.word_encoder(batch['word'], batch['mask']) # [chunk_num, number_of_tokens, 768]
        batch_emb = self.drop(batch_emb)
        assert batch_emb.size()[:2] == batch['mask'].size()  # torch.Size([chunk_num, 60, 768]) torch.Size([batch_size, 60])
        proto = self.__get_proto__(batch_emb, batch['label'], batch['text_mask'], max_label)

        self.proto_batch_list.append(proto)

    
    def compute_prototypes(self):

        self.prototypes = torch.mean(torch.stack(self.proto_batch_list), dim=0)
        self.proto_batch_list = []



    def predict(self, batch):

        batch_emb = self.word_encoder(batch['word'], batch['mask']) # [num_sent, number_of_tokens, 768]
        batch_emb = self.drop(batch_emb)

        # Prototypical Networks
        assert batch_emb.size()[:2] == batch['mask'].size()

        batch_dist = self.__batch_dist__(self.prototypes, batch_emb, batch['text_mask']) # [num_of_query_tokens, class_num]
        batch_nearest_dist, pred = torch.max(batch_dist, 1)  ## (number of query tokens, ), (number of query tokens, )

        return batch_nearest_dist, pred


    
    def forward(self, support, query):
        '''
        support: support set.
        query: query set.
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]  ## number of chunks in instance i
            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return logits, pred

    def loss(self, logits, label):
        N = logits.size(-1)
        if self.proto_loss_mode == 'normal':
            return self.cost(logits.view(-1, N), label.view(-1))
        elif self.proto_loss_mode == 'unlikelihood':
            un_logits = logits[label != -1]
            un_label = label[label != -1]

            lbl_idx_lkup = nn.Embedding.from_pretrained(torch.eye(N)).cuda()
            with torch.no_grad():
                lkup_lbl = lbl_idx_lkup(un_label)
            reshape_lbl = lkup_lbl.reshape(-1)
            del lbl_idx_lkup

            reshape_logits = un_logits.reshape(-1)

            loss = nn.BCEWithLogitsLoss()(reshape_logits, reshape_lbl)
            return loss
        else:
            raise NotImplementedError('Invalid proto_loss_mode!')

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __delete_ignore_index(self, pred, label):
        pred = pred[label != self.ignore_index]
        label = label[label != self.ignore_index]
        assert pred.shape[0] == label.shape[0]
        return pred, label

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        pred, label = self.__delete_ignore_index(pred, label)
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


    def __dist__(self, x, y, dim):

        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_proto__(self, embedding, label, text_mask, max_label=None):
        ## input batch * length of chunks (60) * 768
        proto = []
        embedding = embedding[text_mask==1].view(-1, embedding.size(-1))  ## (input batch * length of chunks - |text_mask==0|, 768)
        label = torch.cat(label, 0)
        assert label.size(0) == embedding.size(0)
        label_set = set(torch.unique(label).tolist())
        if not max_label:
            max_label = torch.max(label)
        for l in range(max_label+1):
            if l in label_set:
                proto.append(torch.mean(embedding[label==l], 0))
            else:
                proto.append(torch.zeros(embedding.size(-1)).cuda())
        proto = torch.stack(proto)
        return proto

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

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
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
    def __transform_label_to_tag__(self, pred, query):
        '''
        flatten labels and transform them to string tags
        '''
        pred_tag = []
        label_tag = []
        current_sent_idx = 0 # record sentence index in the batch data
        current_token_idx = 0 # record token index in the batch data
        assert len(query['sentence_num']) == len(query['label2tag'])

        # iterate by each query set
        for idx, num in enumerate(query['sentence_num']):
            true_label = torch.cat(query['label'][current_sent_idx:current_sent_idx+num], 0)  ## get a labels of a SENTENCE
            # drop ignore index
            true_label = true_label[true_label!=self.ignore_index]  ## get rid of -1 labels of a SENTENCE
            
            true_label = true_label.cpu().numpy().tolist()
            set_token_length = len(true_label)
            # use the idx-th label2tag dict
            pred_tag += [query['label2tag'][idx][label] for label in pred[current_token_idx:current_token_idx + set_token_length]]
            label_tag += [query['label2tag'][idx][label] for label in true_label]
            # update sentence and token index
            current_sent_idx += num
            current_token_idx += set_token_length
        assert len(pred_tag) == len(label_tag)
        assert len(pred_tag) == len(pred)
        return pred_tag, label_tag

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)  # label name: [(start, end), (start, end)]
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span

    def get_tag_by_label(self, pred, label, query):
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        return pred_tag, label_tag

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def metrics_by_entity(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        pred_class_span = self.__get_class_span_dict__(pred)
        label_class_span = self.__get_class_span_dict__(label)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    # The following snippet is from FewNERD
    #    (https://github.com/thunlp/Few-NERD/)
    # This source code is licensed under the Apache 2.0 license,
    # found in the 3rd-party-licenses.txt file in the root directory of this source tree.
    def error_analysis(self, pred, label, query):
        '''
        return 
        token level false positive rate and false negative rate
        entity level within error and outer error 
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        fp = torch.sum(((pred > 0) & (label == 0)).type(torch.FloatTensor))
        fn = torch.sum(((pred == 0) & (label > 0)).type(torch.FloatTensor))
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        within, outer, total_span = self.__get_type_error__(pred, label, query)
        return fp, fn, len(pred), within, outer, total_span
