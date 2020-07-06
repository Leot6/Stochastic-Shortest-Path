"""
efficiently compute the approximated stochastic shortest path - precomputing part
"""

import time
import copy
import math
import numpy as np
import networkx as nx
from tqdm import tqdm

from config import NETWORK
NETWORK_STOCHASTIC = copy.deepcopy(NETWORK)


# given a path, return its variance
def get_path_var(path):
    var = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        var += NETWORK.get_edge_data(u, v, default={'var': None})['var']
    return round(var, 2)


# find the smallest possible nonzero mean and variance of a single edge in the graph
def find_the_smallest_mean_and_var():
    print('finding the smallest mean and var ...')
    smallest_mean = np.inf
    smallest_var = np.inf
    for u, v in NETWORK.edges():
        dur = NETWORK.get_edge_data(u, v, default={'dur': None})['dur']
        var = NETWORK.get_edge_data(u, v, default={'var': None})['var']
        if dur < smallest_mean:
            smallest_mean = dur
        if var < smallest_var:
            smallest_var = var
    print('     smallest_mean:', smallest_mean)
    print('     smallest_var:', smallest_var)
    return round(smallest_mean, 2), round(smallest_var, 2)


# find the largest possible variance of any path in the graph (takes around 10 min)
def find_the_largest_var_of_any_path():
    print('finding the largest var of any path... ')
    largest_var_path = None
    largest_var = 0
    print('     computing all shortest paths...')
    len_path = dict(nx.all_pairs_dijkstra(NETWORK, weight='dur'))
    nodes_id = list(range(1, NETWORK.number_of_nodes() + 1))
    for o in tqdm(nodes_id, desc='checking all paths'):
        for d in tqdm(nodes_id):
            try:
                path = len_path[o][1][d]
                var = get_path_var(path)
                if var > largest_var:
                    largest_var_path = path
                    largest_var = var
            except nx.NetworkXNoPath:
                print('no path between', o, d)
    print('     largest_var', largest_var, 'path_length:', len(largest_var_path))
    print('     longerst_path:', largest_var_path)
    return largest_var

    # found path: mean=1367.58, var=737.71, num of edges = 46
    # path = [1917, 1898, 1855, 1912, 2034, 2074, 2095, 2107, 2164, 2286, 2392, 2378, 2365, 2349, 2332, 2314, 2291,
    #         2268, 2249, 2230, 2473, 2529, 2596, 2618, 2635, 2659, 2681, 2705, 2718, 2739, 2754, 2771, 2790, 2855,
    #         2862, 2886, 2919, 2982, 3053, 3159, 3277, 3302, 3354, 3419, 3384, 3343]


# Mean-Risk Model
def compute_parameter_set_k_mean_risk(epsilon=0.5, smallest_var=0.2, largest_var=705):
    print('')
    xi = math.sqrt(epsilon / (1 + epsilon))
    lower_bound = smallest_var
    uper_bound = largest_var
    K = []
    new_k = round(lower_bound, 4)
    i = 0
    while new_k < uper_bound:
        K.append(new_k)
        i += 1
        new_k = round((1 + xi)**i * lower_bound, 4)
    K.append(uper_bound)
    print('num of k (mean risk):', len(K), K)
    print(K)
    return K


# Probability-Tail Model
# the value of ddl_lb is not specified in the paper, here we are using an arbitrary value
def compute_parameter_set_k_probability(epsilon=0.5, ddl_lb=1.5, smallest_mean=5, smallest_var=0.2, largest_var=705):
    xi = epsilon / 2
    lower_bound = 2 * smallest_var / ddl_lb
    uper_bound = round(2 * largest_var / (epsilon * smallest_mean), 2)
    print('uper_bound', uper_bound)
    K = []
    new_k = round(lower_bound, 4)
    i = 0
    while new_k < uper_bound:
        K.append(new_k)
        i += 1
        new_k = round((1 + xi)**i * lower_bound, 4)
    K.append(uper_bound)
    print('num of k:', len(K), K)
    print(K)
    return K


if __name__ == '__main__':
    start_time = time.time()
    epsilon = 0.5
    ddl_lb = 1.5
    min_mean, min_var = find_the_smallest_mean_and_var()
    max_var = find_the_largest_var_of_any_path()

    K_mean_risk = compute_parameter_set_k_mean_risk(epsilon, min_var, max_var)
    K_probability = compute_parameter_set_k_probability(epsilon, ddl_lb, min_mean, min_var, max_var)
    # K = [0, 0.2, 0.3155, 0.4976, 0.7849, 1.2381, 1.9529, 3.0803, 4.8588, 7.664, 12.0888, 19.0683, 30.0774, 47.4425,
    #      74.8335, 118.0386, 186.1882, 293.684, 463.2426, 705, np.inf]

    print('...running time : %.05f seconds' % (time.time() - start_time))
