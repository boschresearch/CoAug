#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#
#

from tallor.label_functions.LF_template import *
from tallor.rule_kits.rule_reader import surface_reader

class NCBI_SurfaceForm(SurfaceForm):
    def __init__(self, ner_label):

        dictionary = surface_reader('ncbi_disease', number=20)  ## only start from positive surface name rule
        negative_set = set()
        
        super().__init__(ner_label, dictionary, negative_set)



class NCBI_Prefix(Prefix):
    def __init__(self, ner_label):

        prefix_dict = dict()
        
        neg_prefix_set = set()

        super().__init__(ner_label, prefix_dict, neg_prefix_set)

        
class NCBI_Suffix(Suffix):
    def __init__(self, ner_label):

        suffix_dict = dict()

        neg_suffix_set = set()

        super().__init__(ner_label, suffix_dict, neg_suffix_set)

class NCBI_InclusivePreNgram(InclusivePreNgram):

    def __init__(self, ner_label):

        inclusive_pre_dict = dict()
        
        neg_inclusive_pre_set = set()

        super().__init__(ner_label, inclusive_pre_dict, neg_inclusive_pre_set)


class NCBI_InclusivePostNgram(InclusivePostNgram):

    def __init__(self, ner_label):

        inclusive_post_dict = dict()
        neg_inclusive_post_set = set()
        super().__init__(ner_label, inclusive_post_dict, neg_inclusive_post_set)


class NCBI_ExclusivePreNgram(ExclusivePreNgram):

    def __init__(self, ner_label):

        exclusive_pre_dict = dict()
        
        neg_exclusive_pre_set = set()

        super().__init__(ner_label, exclusive_pre_dict, neg_exclusive_pre_set)


class NCBI_ExclusivePostNgram(ExclusivePostNgram):

    def __init__(self, ner_label):

        exclusive_post_dict = dict()

        neg_exclusive_post_set = set()

        super().__init__(ner_label, exclusive_post_dict, neg_exclusive_post_set)

class NCBI_PosTagRule(PosTagRule):

    def __init__(self):

        POS_set = {'NUM':'Entity'}
        neg_POS_set = {}
        super().__init__(POS_set, neg_POS_set)

class NCBI_CapitalRule(CapitalRule):

    def __init__(self):

        Capitalized = {'capitalized', 'upper'}
        super().__init__(Capitalized)
        
class NCBI_DependencyRule(DependencyRule):

    def __init__(self):

        Dep_dict = dict()
        neg_Dep_set = set()
        super().__init__(Dep_dict, neg_Dep_set)

class NCBI_ComposedRule(ComposedRule):

    def __init__(self, ner_label):

        ExPre = NCBI_ExclusivePreNgram(ner_label)
        ExPost = NCBI_ExclusivePostNgram(ner_label)
        POStag = NCBI_PosTagRule()
        DepRule = NCBI_DependencyRule()
        composed_rule = [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]

        #1. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]
        #2. [(ExPre, ExPost), (DepRule, POStag)]
        #3. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost)]
        #4. [(ExPre, ExPost)]
        super().__init__(ner_label, composed_rule)


