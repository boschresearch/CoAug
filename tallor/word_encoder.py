#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#

import torch
import torch.nn as nn
from transformers import BertModel

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path='bert-base-uncased'): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True, return_dict=True)
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        word_embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768], squeeze (add on) the 1st dimension
        return word_embeddings
