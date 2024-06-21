#
#  This source code is from TALLOR
#    (https://github.com/JiachengLi1995/TALLOR)
#  This source code is licensed under the AGPL-3.0 license,
#  found in  the 3rd-party-licenses.txt file in the root directory of this source tree.
#

def surface_reader(dataset, number = 20):

    res = dict()
    with open('./data/'+dataset+f'/selected_dict_filted.txt') as dict_f:
        for line in dict_f.readlines():
            line = line.strip().split('\t')
            label, term = line[0], line[1]
            if len(term)>3:
                res[term.lower()] = label
            
            if len(res)>number:
                break

    return res