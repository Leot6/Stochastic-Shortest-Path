"""
compute the exact path that maximize the probability of reaching a destination within a particular travel deadline.
"""

import time
import pickle
import copy
import math
import numpy as np
import pandas as pd
import networkx as nx

with open('NYC_NET.pickle', 'rb') as f:
    NYC_NET = pickle.load(f)
G = copy.deepcopy(NYC_NET)


def get_the_minimum_distance_path(graph, source, target):
    path = nx.shortest_path(graph, source, target, weight='dist')
    return path


def get_the_minimum_distance_path_length(graph, source, target):
    distance = nx.shortest_path_length(graph, source, target, weight='dist')
    return round(distance, 2)


def get_the_minimum_duration_path(graph, source, target):
    path = nx.shortest_path(graph, source, target, weight='dur')
    return path


def get_the_minimum_duration_path_length(graph, source, target):
    duration = nx.shortest_path_length(graph, source, target, weight='dur')
    return round(duration, 2)


def get_path_mean_and_var(path):
    mean = 0.0
    var = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        mean += NYC_NET.get_edge_data(u, v, default={'dur': None})['dur']
        var += NYC_NET.get_edge_data(u, v, default={'var': None})['var']
    return round(mean, 2), round(var, 2)


def get_lemada_optimal_path(lemada, onid, dnid):
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
    path = get_the_minimum_duration_path(G, onid, dnid)
    mean, var = get_path_mean_and_var(path)
    return path, mean, var


# the cumulative distribution function (CDF) of the standard normal distribution
def get_path_phi(d, m, v):
    return round((d-m)/(math.sqrt(v)), 2)


# compute path that maximize the probability of arriving at a destination before a given time deadline
def stochastic_shortest_path(d, onid, dnid):
    """
    Attributes:
        d: deadline
        onid: origin node id
        dnid: destination node id
        m: mean
        v: variance
        l:left
        r:right
    """
    best_path = []
    phi_best = 0
    candidate_regions = []
    path_0, m_0, v_0 = get_lemada_optimal_path(0, onid, dnid)
    path_inf, m_inf, v_inf = get_lemada_optimal_path(np.inf, onid, dnid)

    if path_0 == path_inf:
        # print('Case 1: path_0 == path_inf !!!')
        return path_0

    phi_0 = get_path_phi(d, m_0, v_0)
    phi_inf = get_path_phi(d, m_inf, v_inf)
    # print('m_0', m_0, 'v_0', v_0, 'phi_path_0', phi_0)
    # print('m_inf', m_inf, 'v_inf', v_inf, 'phi_path_inf', phi_inf)
    if phi_0 < phi_inf:
        print('Case 2: path_inf is better than path_0 !!!')
        quit()

    # region: (left path, right path)
    candidate_regions.append(((m_0, v_0), (m_inf, v_inf)))

    while len(candidate_regions) != 0:
        region = candidate_regions.pop()
        # print('current region', region)
        (m_l, v_l), (m_r, v_r) = region
        # print('m_l, v_l, m_r, v_r', m_l, v_l, m_r, v_r)
        phi_probe = get_path_phi(d, m_l, v_r)
        # print('phi_probe', phi_probe)

        if phi_probe < phi_best:
            continue
        lemada = - (m_l - m_r) / (v_l - v_r)
        path, m, v = get_lemada_optimal_path(lemada, onid, dnid)
        phi_path = get_path_phi(d, m, v)
        if (m == m_l and v == v_l) or (m == m_r and v == v_r):
            continue
        if phi_path > phi_best:
            best_path = path
            phi_best = phi_path
        phi_probe_l = get_path_phi(d, m_l, v)
        phi_probe_r = get_path_phi(d, m, v_r)
        if phi_probe_l > phi_best:
            candidate_regions.append(((m_l, v_l), (m, v)))
        if phi_probe_r > phi_best:
            candidate_regions.append(((m, v), (m_r, v_r)))

        # print('lemada', lemada, 'm', m, 'v', v, 'phi_path', phi_path)

    return best_path


if __name__ == '__main__':
    onid = 12
    dnid = 3788
    d = get_the_minimum_duration_path_length(NYC_NET, onid, dnid) * 1.2

    start_time = time.time()

    best_path = stochastic_shortest_path(d, onid, dnid)
    m_best, v_best = get_path_mean_and_var(best_path)
    phi_best = get_path_phi(d, m_best, v_best)
    # print('m_best, v_best, phi_best', m_best, v_best, phi_best)
    # print('best_path', best_path)

    print('...running time : %.05f seconds' % (time.time() - start_time))
