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

from tallor.data_loader import get_loader
from tallor.proto_framework import IEFramework
from tallor.sentence_encoder import BERTSentenceEncoder
from tallor.word_encoder import BERTWordEncoder
from tallor.utils import LabelIdMapper
from models.JointIE import JointIE
from models.proto import Proto
import torch
import sys
import argparse
import os
import logging
import datetime
import numpy as np
import random

BASE_EXP_DIR = 'exp_out'

def set_seed(seed=0):  # default seed is 0
    torch.manual_seed(seed)  # returns a torch.Generator object, set seed for cpu
    torch.cuda.manual_seed_all(seed)  # set seed for all gpus
    np.random.seed(seed)  # set seed for numpy
    random.seed(seed)  # set seed for python
    torch.backends.cudnn.deterministic = True  # fix cuda random seed

def main():

    parser = argparse.ArgumentParser()
    ## File parameters
    parser.add_argument('--train', default='train',
            help='train file')
    parser.add_argument('--val', default='dev',
            help='val file')
    parser.add_argument('--test', default='test',
            help='test file')
    parser.add_argument('--root', default='./data',
            help='dataset root')
    parser.add_argument('--dataset', default='bc5cdr',
            help='dataset')

    ## seed
    parser.add_argument('--seed', default=0, type=int,
            help='seed for experiment.')

    ## span
    parser.add_argument('--max_span_width', default=5, type=int,
            help='max number of word in a span')
    

    ## encoder
    parser.add_argument('--lexical_dropout', default=0.5, type=float,
            help='Embedding dropout')
    parser.add_argument('--embedding_size', default=768, type=float,
            help='Embedding dimension')
    parser.add_argument('--lower', default=1, type=int,
            help='1: lower case  0: upper case')
    parser.add_argument('--freeze', action='store_true',
            help='freeze bert model')
    
    ## model
    parser.add_argument('--model', default='JointIE',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: bert or scibert')
    parser.add_argument('--hidden_size', default=512, type=int,
           help='hidden size')
    parser.add_argument('--context_layer', default=1, type=int,
           help='number of contextual layers')
    parser.add_argument('--context_dropout', default=0, type=int,
           help='dropout rate in the contextual layer')
    parser.add_argument('--dropout', default=0.3, type=float,
           help='dropout rate')
    parser.add_argument('--span_width_dim', default=64, type=int,
           help='dimension of embedding for span width')
    parser.add_argument('--spans_per_word', default=0.6, type=float,
           help='thershold number of spans in each sentence')

    ## Train
    parser.add_argument('--batch_size', default=32, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=50000, type=int,
            help='num of iters in ner model training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in ner model validation')
    parser.add_argument('--test_iter', default=1000, type=int,
            help='num of iters in ner model testing')
    parser.add_argument('--val_step', default=1500, type=int,
           help='val after training how many iters')
    parser.add_argument('--warmup_step', default=100, type=int,
           help='warmup steps for neural tagger')
    parser.add_argument('--update_threshold', default=0.7, type=float,
           help='the ratio of the most confident data used for evaluateing and updating new rules')
    parser.add_argument('--labeled_ratio', default=0, type=float,
           help='The ratio of labeled data used for training.')
    parser.add_argument('--not_use_soft_label', action='store_false',
           help='Do not use soft label for training.')

    ## Rules
    parser.add_argument('--rule_threshold', default=2, type=int,
            help='Rule frequency threshold.')
    parser.add_argument('--ap_threshold', default=0.75, type=float,
        help='AutoPhrase threshold.')
    parser.add_argument('--rule_topk', default=20, type=int,
            help='Select topk rules added to rule set.')

    ## Instance Selector
    parser.add_argument('--global_sample_times', default=50, type=int,
            help='Sample times for global scores.')
    parser.add_argument('--threshold_sample_times', default=100, type=int,
            help='Sample times for computing dynamic threshold.')
    parser.add_argument('--temperature', default=0.8, type=float,
            help='Temperature to control threshold.')

    ## Fewshot
    parser.add_argument('--trainN', default=2, type=int,
            help='N in train')
    parser.add_argument('--N', default=2, type=int,
            help='N way')
    parser.add_argument('--K', default=2, type=int,
            help='K shot')
    parser.add_argument('--Q', default=2, type=int,
            help='Num of query per class')
    parser.add_argument('--fewshot_batch_size', default=16, type=int,
            help='batch size for few show setting')
    parser.add_argument('--fewnerd_data_mode', default='inter', type=str,
            help='inter or intra')
    parser.add_argument('--fewshot_lr', default=1e-4, type=float,
            help='few shot learning rate')
    parser.add_argument('--fewshot_train_iter', default=12000, type=int,
            help='num of iters in few shot training')
    parser.add_argument('--fewshot_val_iter', default=500, type=int,
            help='num of iters in few shot validation')
    parser.add_argument('--fewshot_test_iter', default=500, type=int,
            help='num of iters in few shot testing')
    parser.add_argument('--fewshot_val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--encoder_model', default='bert-base-uncased',
            help='encoder model class')
    parser.add_argument('--proto_loss_mode', default='normal', type=str,
            help='proto loss mode')
    parser.add_argument('--constraint_threshold', default=0.98, type=float,
           help='threshold for constraint optimization.')

    ## Save
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--load_fewshot_ckpt', default=None,
           help='load fewshot_model ckpt')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    # Load
    parser.add_argument('--load_dataset_path', default=None,
           help='Loading path for datasets from incomplete run.')

    parser.add_argument('--mode', default='coaug', type=str,
           help='tallor, proto, or coaug', choices=['tallor', 'proto', 'coaug'])
        
    opt = parser.parse_args()
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    opt.lower = bool(opt.lower)
    root = os.path.join(opt.root, opt.dataset)
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))

    # Set seeds for experiments
    set_seed(opt.seed)

    ## Create experiment checkpointing locations
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second, now.microsecond)
    exp_dir = os.path.join(BASE_EXP_DIR, opt.dataset, opt.mode, 'ProtoBERT', str(opt.seed), ts)
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    ## set sentence encoder for neural tagger       
    if encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        opt.embedding_size = 768
        opt.lower = 'uncased' in pretrain_ckpt
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, opt.lower)
    elif encoder_name == 'scibert':
        pretrain_ckpt = opt.pretrain_ckpt or 'allenai/scibert_scivocab_uncased'
        opt.embedding_size = 768
        opt.lower = 'uncased' in pretrain_ckpt
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, opt.lower)
    elif encoder_name == 'biobert':
        pretrain_ckpt = opt.pretrain_ckpt or 'dmis-lab/biobert-v1.1'
        opt.embedding_size = 768
        opt.lower = 'uncased' in pretrain_ckpt
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, opt.lower)
    else:
        raise NotImplementedError

    if opt.freeze:
        sentence_encoder.freeze()
    
    label_id_mapper = LabelIdMapper()

    ## read dataset
    dataset_name = os.path.basename(root)  # get the local name of the dataset
    print(dataset_name)
    train_data_loader, unlabel_data_loader, _ = get_loader(root, opt.train, sentence_encoder, batch_size, label_id_mapper, is_train=True, opt=opt)
    val_data_loader, val_dataset = get_loader(root, opt.val, sentence_encoder, batch_size, label_id_mapper, is_train=False, opt=opt)
    test_data_loader, _ = get_loader(root, opt.test, sentence_encoder, batch_size, label_id_mapper, is_train=False, opt=opt)

    ## set logger
    prefix = model_name
    fewshot_prefix = 'ProtoBERT'
    logger = set_logger(exp_dir, prefix)
    logger.info(opt)

    fewshot_batch_size = opt.fewshot_batch_size

    framework = IEFramework(label_id_mapper, dataset_name, opt, logger, batch_size, fewshot_batch_size,
                                train_data_loader, val_data_loader, test_data_loader, unlabel_data_loader, val_dataset,
                                mode=opt.mode, exp_dir=exp_dir)
        
    if model_name == 'JointIE':
        model = JointIE(sentence_encoder, opt.hidden_size, opt.embedding_size, label_id_mapper, 
                        opt.context_layer, opt.context_dropout, opt.dropout,
                        max_span_width=opt.max_span_width, span_width_embedding_dim=opt.span_width_dim,
                        spans_per_word=opt.spans_per_word, use_soft_label=opt.not_use_soft_label)
        fewshot_model = Proto(BERTWordEncoder(opt.encoder_model), proto_loss_mode=opt.proto_loss_mode)
        if torch.cuda.is_available():
            model.cuda()
            fewshot_model.cuda()
    else:
        raise NotImplementedError
    
    if not os.path.exists(f'{exp_dir}/checkpoint'):
        os.mkdir(f'{exp_dir}/checkpoint')
    ckpt = f'{exp_dir}/checkpoint/{prefix}.pth.tar'
    fewshot_ckpt = f'{exp_dir}/checkpoint/{fewshot_prefix}.pth.tar'
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not opt.only_test:
        framework.train(model, prefix, fewshot_model, fewshot_prefix, load_ckpt=opt.load_ckpt, save_ckpt=ckpt, fewshot_save_ckpt=fewshot_ckpt, val_step=opt.val_step, 
                        train_iter=opt.train_iter, val_iter=opt.val_iter, warmup_step=opt.warmup_step, update_threshold=opt.update_threshold)
    else:
        ckpt = opt.load_ckpt
        fewshot_ckpt = opt.load_fewshot_ckpt

    if opt.mode != 'proto':
        ner_f1, precision, recall = framework.eval(model, opt.test_iter, -1, ckpt=ckpt)
    else:
        precision, recall, ner_f1, _, _, _, _ = framework.fewshot_labeler.eval(fewshot_model, opt.test_iter, ckpt=fewshot_ckpt)
        print('STEP:-1. Current performance on NER: F1: {0:2.4f}, P: {1:2.4f}, R: {2:2.4f}.'.format(ner_f1, precision, recall))

def set_logger(exp_dir, prefix):
    ## set logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('WeakNER')

    if not os.path.exists(f'{exp_dir}/logging'):
        os.mkdir(f'{exp_dir}/logging')

    file_handler = logging.FileHandler(f'{exp_dir}/logging/'+prefix+'.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger

if __name__ == "__main__":
    main()
