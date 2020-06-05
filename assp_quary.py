"""
efficiently compute the approximated stochastic shortest path - precomputing part
"""

import time
import pickle
import copy
import numpy as np
from ssp import get_lemada_optimal_path, get_path_phi, get_the_minimum_duration_path_length, get_path_mean_and_var

with open('NYC_NET.pickle', 'rb') as f:
    NYC_NET = pickle.load(f)
G = copy.deepcopy(NYC_NET)


# compute path that maximize the probability of arriving at a destination before a given time deadline
def approximated_stochastic_shortest_path(k, d, onid, dnid):
    """
    Attributes:
        k:
        d: deadline
        onid: origin node id
        dnid: destination node id
        m: mean
        v: variance
    """

    path_0, m_0, v_0 = get_lemada_optimal_path(0, onid, dnid)
    phi_0 = get_path_phi(d, m_0, v_0)
    path_inf, m_inf, v_inf = get_lemada_optimal_path(np.inf, onid, dnid)
    phi_inf = get_path_phi(d, m_inf, v_inf)
    if path_0 == path_inf:
        return path_0
    elif phi_0 > phi_inf:
        best_path = path_0
        phi_best = phi_0
    else:
        best_path = path_inf
        phi_best = phi_inf

    for lemada in k:
        path, m, v = get_lemada_optimal_path(lemada, onid, dnid)
        phi_path = get_path_phi(d, m, v)
        if phi_path > phi_best:
            best_path = path
            phi_best = phi_path
    return best_path


if __name__ == '__main__':
    onid = 2
    dnid = 1644
    k = [0.2667, 0.3333, 0.4167, 0.5208, 0.651, 0.8138, 1.0173, 1.2716, 1.5895, 1.9868, 2.4835, 3.1044, 3.8805,
         4.8506, 6.0633, 7.5791, 9.4739, 11.8424, 14.803, 18.5037, 23.1296, 28.9121, 36.1401, 45.1751, 56.4689,
         70.5861, 88.2326, 110.2907, 137.8634, 172.3293, 215.4116, 269.2645, 336.5807, 420.7258, 525.9073, 564.0]

    d = get_the_minimum_duration_path_length(NYC_NET, onid, dnid) * 1.2

    start_time = time.time()

    best_path = approximated_stochastic_shortest_path(k, d, onid, dnid)
    m_best, v_best = get_path_mean_and_var(best_path)
    # print('m_best, v_best', best_path, m_best, v_best)
    phi_best = get_path_phi(d, m_best, v_best)
    print('m_best, v_best, phi_best', m_best, v_best, phi_best)
    # print('best_path', best_path)

    print('...running time : %.05f seconds' % (time.time() - start_time))
