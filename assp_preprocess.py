"""
efficiently compute the approximated stochastic shortest path - precomputing part
"""

import time
import copy
import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from graph import NYC_NET
from ssp import get_path_mean_and_var

G = copy.deepcopy(NYC_NET)
NYC_NOD = pd.read_csv('./graph/nodes.csv')


# given a path, return its variance
def get_path_var(path):
    var = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        var += NYC_NET.get_edge_data(u, v, default={'var': None})['var']
    return round(var, 2)


# find the smallest possible nonzero mean and variance of a single edge in the graph
def find_the_smallest_mean_and_var():
    smallest_mean = np.inf
    smallest_var = np.inf
    for u, v in NYC_NET.edges():
        dur = NYC_NET.get_edge_data(u, v, default={'dur': None})['dur']
        var = NYC_NET.get_edge_data(u, v, default={'var': None})['var']
        if dur < smallest_mean:
            smallest_mean = dur
        if var < smallest_var:
            smallest_var = var
    print('smallest_mean:', smallest_mean)
    print('smallest_var:', smallest_var)
    # return round(smallest_mean, 2), round(smallest_var, 2)


# find the largest possible variance of any path in the graph (takes around 10 min)
def find_the_largest_var_of_any_path():
    largest_var_path = None
    largest_var = 0
    len_path = dict(nx.all_pairs_dijkstra(NYC_NET, weight='dur'))
    nodes = pd.read_csv('./graph/nodes.csv')
    nodes_id = list(range(1, nodes.shape[0] + 1))
    for o in tqdm(nodes_id, desc='all paths'):
        for d in tqdm(nodes_id):
            try:
                path = len_path[o][1][d]
                var = get_path_var(path)
                if var > largest_var:
                    largest_var_path = path
                    largest_var = var
            except nx.NetworkXNoPath:
                print('no path between', o, d)
    print('largest_var', largest_var, 'path_length:', len(largest_var_path))
    print('longerst_path:', largest_var_path)

    # found path: mean=2730.52, var=705.21
    # path = [3899, 3914, 3864, 3886, 3890, 3845, 3836, 3917, 3930, 3940, 3964, 3980, 3990, 3993, 3998, 4001, 4009,
    #         3840, 3822, 3788, 3810, 3811, 3787, 3792, 3877, 3872, 3868, 3860, 3850, 3837, 3823, 3806, 3784, 3763,
    #         3740, 3718, 3696, 3677, 3654, 3629, 3576, 3550, 3518, 3492, 3470, 3445, 3395, 3375, 3350, 3327, 3296,
    #         3272, 3248, 3219, 3184, 3159, 3133, 3110, 3090, 3070, 3185, 3295, 3268, 3246, 3218, 3198, 3182, 3156,
    #         3130, 3109, 3084, 3066, 3044, 3022, 3001, 2979, 2961, 2943, 2923, 2895, 2865, 2848, 2831, 2811, 2789,
    #         2768, 2753, 2737, 2716, 2702, 2677, 2717, 2670, 2648, 2643, 2619, 2597, 2509, 2433, 2287, 2255, 2147,
    #         2145, 2129, 2117, 2101, 2073, 2029, 1992, 1962, 1968, 2102, 2120, 1981, 1961, 1955, 1953, 1937, 1870,
    #         1858, 1823, 1733, 1659, 1461, 1396, 1406, 1314, 1160, 1144, 1041, 1034, 941, 839, 824, 830, 844, 944, 951]


# Mean-Risk Model
def compute_parameter_set_k_mean_risk():
    epsilon = 0.5
    smallest_var = 0.2
    largest_var = 705
    xi = math.sqrt(epsilon/(1+epsilon))
    lower_bound = smallest_var
    uper_bound = largest_var
    k = []
    new_k = round(lower_bound, 4)
    i = 0
    while new_k < uper_bound:
        k.append(new_k)
        i += 1
        new_k = round((1 + xi)**i * lower_bound, 4)
    k.append(uper_bound)
    print('num of k:', len(k), k)
    return k


# Probability-Tail Model
def compute_parameter_set_k_probability():
    epsilon = 0.5
    smallest_mean = 5
    smallest_var = 0.2
    largest_var = 705
    d_l = 1.5
    xi = epsilon / 2
    lower_bound = 2 * smallest_var / d_l
    uper_bound = round(2 * largest_var / (epsilon * smallest_mean), 2)
    print('uper_bound', uper_bound)
    k = []
    new_k = round(lower_bound, 4)
    i = 0
    while new_k < uper_bound:
        k.append(new_k)
        i += 1
        new_k = round((1 + xi)**i * lower_bound, 4)
    k.append(uper_bound)
    print('num of k:', len(k), k)
    return k


# def compute_shortest_time_table(len_paths, nodes=NYC_NOD):
#     # nodes = pd.read_csv('nodes.csv')
#     nodes_id = list(range(1, nodes.shape[0] + 1))
#     num_nodes = len(nodes_id)
#     shortest_time_table = pd.DataFrame(-np.ones((num_nodes, num_nodes)), index=nodes_id, columns=nodes_id)
#     for o in tqdm(nodes_id, desc='time-table'):
#         for d in tqdm(nodes_id):
#             try:
#                 duration = round(len_paths[o][0][d], 2)
#                 shortest_time_table.iloc[o - 1, d - 1] = duration
#             except nx.NetworkXNoPath:
#                 print('no path between', o, d)
#     # shortest_time_table.to_csv('time-table-new.csv')
#     return shortest_time_table
#
#
# def compute_shortest_path_table(len_paths, nodes=NYC_NOD):
#     # nodes = pd.read_csv('nodes.csv')
#     nodes_id = list(range(1, nodes.shape[0] + 1))
#     num_nodes = len(nodes_id)
#     shortest_path_table = pd.DataFrame(-np.ones((num_nodes, num_nodes), dtype=int), index=nodes_id, columns=nodes_id)
#     for o in tqdm(nodes_id, desc='path-table'):
#         for d in tqdm(nodes_id):
#             try:
#                 path = len_paths[o][1][d]
#                 if len(path) == 1:
#                     continue
#                 else:
#                     pre_node = path[-2]
#                     shortest_path_table.iloc[o - 1, d - 1] = pre_node
#             except nx.NetworkXNoPath:
#                 print('no path between', o, d)
#     # shortest_path_table.to_csv('path-table-new.csv')
#     return shortest_path_table


def compute_tables(len_paths, nodes=NYC_NOD):
    # nodes = pd.read_csv('nodes.csv')
    nodes_id = list(range(1, nodes.shape[0] + 1))
    num_nodes = len(nodes_id)
    lemada_path_table = pd.DataFrame(-np.ones((num_nodes, num_nodes), dtype=int), index=nodes_id, columns=nodes_id)
    mean_table = pd.DataFrame(-np.ones((num_nodes, num_nodes)), index=nodes_id, columns=nodes_id)
    var_table = pd.DataFrame(-np.ones((num_nodes, num_nodes)), index=nodes_id, columns=nodes_id)
    for o in tqdm(nodes_id, desc='path-table row'):
        for d in tqdm(nodes_id, desc='path-table column'):
            try:
                path = len_paths[o][1][d]
                if len(path) == 1:
                    continue
                else:
                    pre_node = path[-2]
                    lemada_path_table.iloc[o - 1, d - 1] = pre_node
                mean, var = get_path_mean_and_var(path)
                mean_table.iloc[o - 1, d - 1] = mean
                var_table.iloc[o - 1, d - 1] = var
            except nx.NetworkXNoPath:
                print('no path between', o, d)
    # shortest_path_table.to_csv('path-table-new.csv')
    return lemada_path_table, mean_table, var_table


def precompute_tables(K):
    for i in tqdm(range(0, len(K)), desc='precomputing tables:'):
        lemada = K[i]
        print(' updating lemada graph')
        for u, v in G.edges():
            dur = NYC_NET.get_edge_data(u, v, default={'dur': None})['dur']
            var = NYC_NET.get_edge_data(u, v, default={'var': None})['var']
            if dur is np.inf:
                print('error: dur is np.inf !!!')
                quit()
            if lemada == np.inf:
                weight = var
            else:
                weight = dur + lemada * var
            G.edges[u, v]['dur'] = weight
        print(' computing all_pairs_dijkstra...')
        len_paths = dict(nx.all_pairs_dijkstra(G, cutoff=None, weight='dur'))
        print(' writing table value...')
        lemada_path_table, mean_table, var_table = compute_tables(len_paths, nodes=NYC_NOD)
        lemada_path_table.to_csv('./precomputed_tables/lemada_path_table_' + str(i) + '.csv')
        mean_table.to_csv('./precomputed_tables/mean_table_' + str(i) + '.csv')
        var_table.to_csv('./precomputed_tables/var_table_' + str(i) + '.csv')


if __name__ == '__main__':
    start_time = time.time()
    # find_the_largest_var_of_any_path()

    # compute_parameter_set_k_mean_risk()
    # compute_parameter_set_k_probability()
    K = [0, 0.2, 0.3155, 0.4976, 0.7849, 1.2381, 1.9529, 3.0803, 4.8588, 7.664, 12.0888, 19.0683, 30.0774, 47.4425,
         74.8335, 118.0386, 186.1882, 293.684, 463.2426, 705, np.inf]
    precompute_tables(K)

    print('...running time : %.05f seconds' % (time.time() - start_time))
