"""
compute the exact path that maximize the probability of reaching a destination within a particular travel deadline.
"""

import time
import copy
import math
import numpy as np
import networkx as nx
import scipy.stats as st
from config import NETWORK, DEADLINE_COE

NETWORK_STOCHATIC = copy.deepcopy(NETWORK)


# find the shortest path from origin to dest in the given network
def get_the_minimum_duration_path(network, origin, dest):
    path = nx.shortest_path(network, origin, dest, weight='dur')
    return path


# find the duration of the shortest path from origin to dest in the given network
def get_the_minimum_duration_path_length(network, origin, dest):
    try:
        duration = nx.shortest_path_length(network, origin, dest, weight='dur')
        return round(duration, 2)
    except nx.NetworkXNoPath:
        print(f'No path between {origin} and {dest}')
        return 0


# given a path, return its mean and variance
def get_path_mean_and_var(path):
    mean = 0.0
    var = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        mean += NETWORK.get_edge_data(u, v, default={'dur': None})['dur']
        var += NETWORK.get_edge_data(u, v, default={'var': None})['var']
    return round(mean, 2), round(var, 2)


# compute the phi of a given path
def get_path_phi(ddl, mean, var):
    return round((ddl-mean)/(math.sqrt(var)), 4)


# the cumulative distribution function (CDF) of the standard normal distribution
def normal_cdf(ddl, path):
    mean, var = get_path_mean_and_var(path)
    return round(st.norm(mean, var).cdf(ddl), 4)


# compute the shortest lambda path and its mean and var
def get_lambda_optimal_path(lambda_value, onid, dnid):
    for u, v in NETWORK_STOCHATIC.edges():
        dur = NETWORK.get_edge_data(u, v, default={'dur': None})['dur']
        var = NETWORK.get_edge_data(u, v, default={'var': None})['var']
        if dur is np.inf:
            print('error: dur is np.inf !!!')
            quit()
        if lambda_value == np.inf:
            weight = var
        else:
            weight = dur + lambda_value * var
        NETWORK_STOCHATIC.edges[u, v]['dur'] = weight
    path = get_the_minimum_duration_path(NETWORK_STOCHATIC, onid, dnid)
    mean, var = get_path_mean_and_var(path)
    return path, mean, var


# compute path that maximize the probability of arriving at a destination before a given time deadline
def stochastic_shortest_path(ddl, onid, dnid):
    """Find the path that maximizes highest probability of arriving before the ddl

        Args:
            ddl: the deadline of travelling form onid to dnid.
            onid: the node id of the origin
            dnid: the node id of the destination

        Returns:
            best_path: the path that has the highest probability of arriving before the ddl
    """

    candidate_regions = []
    path_0, mean_0, var_0 = get_lambda_optimal_path(0, onid, dnid)
    phi_0 = get_path_phi(ddl, mean_0, var_0)
    path_inf, mean_inf, var_inf = get_lambda_optimal_path(np.inf, onid, dnid)
    phi_inf = get_path_phi(ddl, mean_inf, var_inf)

    if path_0 == path_inf:
        return path_0
    elif phi_0 > phi_inf:
        best_path = path_0
        phi_best = phi_0
    else:
        best_path = path_inf
        phi_best = phi_inf

    # print('mean_0', mean_0, 'var_0', var_0, 'phi_path_0', phi_0, 'cdf_0', normal_cdf(ddl, path_0))
    # print('mean_inf', mean_inf, 'var_inf', var_inf, 'phi_path_inf', phi_inf, 'cdf_inf', normal_cdf(ddl, path_inf))
    #
    # print('path_0', path_0)
    # print('path_inf', path_inf)

    # region: (left path, right path)
    candidate_regions.append(((mean_0, var_0), (mean_inf, var_inf)))

    while len(candidate_regions) != 0:
        (mean_left, var_left), (mean_right, var_right) = candidate_regions.pop()
        phi_probe = get_path_phi(ddl, mean_left, var_right)

        if phi_probe < phi_best:
            continue
        lambda_value = - (mean_left - mean_right) / (var_left - var_right)
        path_new, mean_new, var_new = get_lambda_optimal_path(lambda_value, onid, dnid)
        phi_new = get_path_phi(ddl, mean_new, var_new)
        if (mean_new == mean_left and var_new == var_left) or (mean_new == mean_right and var_new == var_right):
            continue
        if phi_new > phi_best:
            best_path = path_new
            phi_best = phi_new
        phi_probe_left = get_path_phi(ddl, mean_left, var_new)
        phi_probe_right = get_path_phi(ddl, mean_new, var_right)
        if phi_probe_left > phi_best:
            candidate_regions.append(((mean_left, var_left), (mean_new, var_new)))
        if phi_probe_right > phi_best:
            candidate_regions.append(((mean_new, var_new), (mean_right, var_right)))

        # print('lambda', lambda_value, 'm', mean_new, 'v', var_new, 'phi_path', phi_new)
        # if lambda_value > 564:
        #     print('case: large lambda!!')
        #     quit()
    return best_path


if __name__ == '__main__':
    onid = 100
    dnid = 2244
    d = get_the_minimum_duration_path_length(NETWORK, onid, dnid) * DEADLINE_COE
    print('deadline', d)

    start_time = time.time()

    best_path = stochastic_shortest_path(d, onid, dnid)
    m_best, v_best = get_path_mean_and_var(best_path)
    # # print('m_best, v_best', best_path, m_best, v_best)
    phi_best = get_path_phi(d, m_best, v_best)
    print('m_best,', m_best, ' v_best,', v_best, ' phi_best', phi_best, 'cdf_best', normal_cdf(d, best_path))

    print('path_best', best_path)

    print('...running time : %.05f seconds' % (time.time() - start_time))
