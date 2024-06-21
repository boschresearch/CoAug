#  Copyright (c) 2023 Robert Bosch GmbH
#  SPDX-License-Identifier: AGPL-3.0
#
#
# The following snippet is derived from FewNERD
#    (https://github.com/thunlp/Few-NERD/)
# This source code is licensed under the Apache 2.0 license,
# found in the 3rd-party-licenses.txt file in the root directory of this source tree.


import random


class FewshotSampleBase:
    '''
    Abstract Class
    DO NOT USE
    Build your own Sample class and inherit from this class
    '''
    def __init__(self):
        self.class_count = {}

    def get_class_count(self):
        '''
        return a dictionary of {class_name:count} in format {any : int}
        '''
        return self.class_count


# this is an iterable object with __next__ and __iter__
class FewshotSampler:
    '''
    sample one support set and one query set
    '''
    def __init__(self, N, K, Q, label_id_mapper, samples, classes=None, random_state=0):
        '''
        N: int, how many types in each set
        K: int, how many instances for each type in support set
        Q: int, how many instances for each type in query set
        samples: List[Sample], Sample class must have `get_class_count` attribute
        classes[Optional]: List[any], all unique classes in samples. If not given, the classes will be got from samples.get_class_count()
        random_state[Optional]: int, the random seed
        '''
        self.K = K
        self.N = N
        self.Q = Q
        self.samples = samples
        self.classes =[tag for tag in label_id_mapper.label2id.keys() if tag != '']
        self.N = min(self.N, len(self.classes))
        random.seed(random_state)

    def __additem__(self, index, set_class):
        class_count = self.samples[index].get_class_count()
        for class_name in class_count:
            if class_name in set_class:
                set_class[class_name] += class_count[class_name]
            else:
                set_class[class_name] = class_count[class_name]

    def __valid_sample__(self, sample, set_class, target_classes):
        threshold = 2 * set_class['k']
        class_count = sample.get_class_count()
        if not class_count:
            return False
        isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                isvalid = False
            elif class_name not in set_class:
                isvalid = True
            elif set_class[class_name] + class_count[class_name] > threshold:
                isvalid = False
            elif set_class[class_name] < set_class['k']:
                isvalid = True
        return isvalid

    def __finish__(self, set_class):
        if len(set_class) < self.N+1:
            return False
        for k in set_class:
            if set_class[k] < set_class['k']:
                return False
        return True 

    def __get_candidates__(self, target_classes):
        return [idx for idx, sample in enumerate(self.samples) if sample.valid(target_classes)]  ## all instances that have and only have classes belonging to target classes

    def __next__(self):
        '''
        randomly sample one support set and one query set
        return:
        target_classes: List[any]
        support_idx: List[int], sample index in support set in samples list
        support_idx: List[int], sample index in query set in samples list
        '''
        support_class = {'k':self.K}  # support set shot count per class
        support_idx = []
        query_class = {'k':self.Q}  # query set shot count per class
        query_idx = []
        target_classes = random.sample(self.classes, self.N)  # sample 5 class names
        candidates = self.__get_candidates__(target_classes)  # get all samples of the target classes, list
        while not candidates:
            target_classes = random.sample(self.classes, self.N)  # keep sampling classes
            candidates = self.__get_candidates__(target_classes)

        # greedy search for support set
        while not self.__finish__(support_class):
            index = random.choice(candidates)  # choose a sample from candidates
            if index not in support_idx:
                if self.__valid_sample__(self.samples[index], support_class, target_classes):
                    self.__additem__(index, support_class)
                    support_idx.append(index)
        # same for query set
        while not self.__finish__(query_class):
            index = random.choice(candidates)
            if index not in query_idx and index not in support_idx:
                if self.__valid_sample__(self.samples[index], query_class, target_classes):
                    self.__additem__(index, query_class)
                    query_idx.append(index)
        return target_classes, support_idx, query_idx

    def __iter__(self):
        return self
    