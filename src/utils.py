import numpy as np


def get_root_nodes(knowledge_components, link_strengths):
    nodes_with_parents = [key for key in link_strengths.keys()]
    root_nodes = [kc for kc in knowledge_components if kc not in nodes_with_parents]
    return root_nodes


def get_leaf_nodes(knowledge_components, link_strengths):
    nodes_with_children = [y for x in [list(link_strengths[key].keys()) for key in link_strengths.keys()] for y in x]
    leaf_nodes = [kc for kc in knowledge_components if kc not in nodes_with_children]
    return leaf_nodes


def determine_node_type(node_name):
    if node_name.startswith('(Z'):
        node_type = 'prerequisite'
    elif node_name.startswith('(T'):
        node_type = 'transition'
    elif node_name.startswith('(L'):
        node_type = 'leak'
    elif node_name.startswith('(O'):
        node_type = "observable"
    else:
        node_type = 'mastery'
    return node_type


def determine_node_timestamp(node_name):
    return int(node_name.split(")", 1)[1])


def truthtable(n):
    """
    Create a boolean truthtable of a given length, that's to say a list of lists of all combination of n booleans
    :param n: the number of elements to be combinated
    :return: a list of boolean lists, the expected truthtable
    """
    if n < 1:
        return [[]]
    subtable = truthtable(n - 1)
    return [row + [v] for row in subtable for v in [False, True]]


def bool_list_to_int(lst):
    return int('0b' + ''.join(['1' if x else '0' for x in lst]), 2) if lst else 0


def int_to_bool_list(num):
    bin_string = format(num, '#010b')
    return [x == '1' for x in bin_string]


def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l


def gibbs_sample_to_em_data(sample, n_eval):
    import pandas as pd
    em_data_bn0 = pd.Series({x: sample.loc[x] for x in sample.index if x.endswith(')0')})
    em_data_2tbn = []
    for i in range(1, n_eval):
        data_elt = {}
        for elt in [x for x in sample.index if x.endswith(f'){i-1}') and not x.startswith('(Z') and not x.startswith('(T') and not x.startswith('(L')]:
            subelt = elt[:-len(f'{i-1}')]
            data_elt[subelt+"0"] = sample.loc[elt]

        for elt in [x for x in sample.index if x.endswith(f'){i}')]:
            subelt = elt[:-len(f'{i}')]
            data_elt[subelt+"t"] = sample.loc[elt]
        em_data_2tbn.append(pd.Series(data_elt))
    return em_data_bn0, em_data_2tbn


def get_max_timestamp(state):
    return max([determine_node_timestamp(n) for n in list(state.keys())]) + 1


def unrolled_sample_to_2tbn_samples(unrolled_sample):
    temp_samples = []
    n_eval = get_max_timestamp(unrolled_sample)
    for i in range(1, n_eval):
        data_elt = {}
        for elt in [x for x in unrolled_sample.keys() if
                    x.endswith(f'){i - 1}') and not x.startswith('(Z') and not x.startswith(
                        '(T') and not x.startswith('(L') and not x.startswith('(O')]:
            subelt = elt[:-len(f'{i - 1}')]
            data_elt[subelt + "0"] = unrolled_sample[elt]

        for elt in [x for x in unrolled_sample.keys() if x.endswith(f'){i}')]:
            subelt = elt[:-len(f'{i}')]
            data_elt[subelt + "t"] = unrolled_sample[elt]
        temp_samples.append(data_elt)
        del data_elt

    return temp_samples


def get_strongest_folds(full, axis="user_id", nb_folds=5):
    from sklearn.model_selection import KFold
    all_elements = full[axis].unique()
    kfold = KFold(nb_folds, shuffle=True)
    folds = []
    for i, (train, test) in enumerate(kfold.split(all_elements)):
        list_of_test_ids = []
        for element_id in test:
            list_of_test_ids += list(full.query(f'{axis} == {all_elements[element_id]}').index)
        folds.append(np.array(list_of_test_ids))

    return folds


def get_graph_from_arcs(tup):
    di = {}
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


#def generate_learner_pool_from_dataset(dataset):
