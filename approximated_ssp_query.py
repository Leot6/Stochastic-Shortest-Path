"""
efficiently compute the approximated stochastic shortest path - query
"""

import time
import copy
import numpy as np
from stochastic_shortest_path import get_lambda_optimal_path, get_path_phi, get_the_minimum_duration_path_length, \
    get_path_mean_and_var
from config import NETWORK, DEADLINE_COE


# compute path that maximize the probability of arriving at a destination before a given time deadline,
# computing on-the-fly
def approximated_stochastic_shortest_path(K, ddl, onid, dnid):
    """Find the approximated path that maximizes highest probability of arriving before the ddl

        Args:
            K: a set of precomputed lambda values
            ddl: the deadline of travelling form onid to dnid.
            onid: the node id of the origin
            dnid: the node id of the destination

        Returns:
            best_path: the path that has the highest probability of arriving before the ddl
    """

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

    for lambda_value in K:
        path_new, mean_new, var_new = get_lambda_optimal_path(lambda_value, onid, dnid)
        phi_new = get_path_phi(ddl, mean_new, var_new)
        if phi_new > phi_best:
            best_path = path_new
            phi_best = phi_new
    return best_path


if __name__ == '__main__':
    onid = 2
    dnid = 1644
    k = [0.2667, 0.3333, 0.4167, 0.5208, 0.651, 0.8138, 1.0173, 1.2716, 1.5895, 1.9868, 2.4835, 3.1044, 3.8805,
         4.8506, 6.0633, 7.5791, 9.4739, 11.8424, 14.803, 18.5037, 23.1296, 28.9121, 36.1401, 45.1751, 56.4689,
         70.5861, 88.2326, 110.2907, 137.8634, 172.3293, 215.4116, 269.2645, 336.5807, 420.7258, 525.9073, 564.0]

    d = get_the_minimum_duration_path_length(NETWORK, onid, dnid) * DEADLINE_COE

    start_time = time.time()

    best_path = approximated_stochastic_shortest_path(k, d, onid, dnid)
    m_best, v_best = get_path_mean_and_var(best_path)
    # print('m_best, v_best', best_path, m_best, v_best)
    phi_best = get_path_phi(d, m_best, v_best)
    print('m_best, v_best, phi_best', m_best, v_best, phi_best)
    # print('best_path', best_path)

    print('...running time : %.05f seconds' % (time.time() - start_time))
