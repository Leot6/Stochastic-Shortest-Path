"""
try to find out the percentage of how many trips would have a different path than the shortest path
"""

import time
import numpy as np
import scipy.stats as st
from tqdm import tqdm

from ssp import get_the_minimum_duration_path, get_the_minimum_duration_path_length, stochastic_shortest_path, \
    get_path_mean_and_var, get_path_phi
from assp_query import approximated_stochastic_shortest_path
from graph import NYC_NET, NOD_LOC, REQ_DATA


# find the nearest node to[lng, lat] in Manhattan network
def find_nearest_node(lng, lat):
    nearest_node_id = None
    d = np.inf
    for nid, nlng, nlat in NOD_LOC:
        # d_ = get_haversine_distance(lng, lat, nlng, nlat)
        d_ = abs(lng-nlng) + abs(lat-nlat)
        if d_ < d:
            d = d_
            nearest_node_id = nid

    if nearest_node_id is None:
        print()
        print('nearest_node_id not found')
        print('coordination', lng, lat)
        print('d', d)
        print()
    return int(nearest_node_id)


def load_trip(req_idx):
    olng = REQ_DATA.iloc[req_idx]['olng']
    olat = REQ_DATA.iloc[req_idx]['olat']
    dlng = REQ_DATA.iloc[req_idx]['dlng']
    dlat = REQ_DATA.iloc[req_idx]['dlat']
    onid = find_nearest_node(olng, olat)
    dnid = find_nearest_node(dlng, dlat)
    return onid, dnid


def find_out_how_many_trips_will_have_a_differnt_path():
    req_idx = 0
    num_different = 0
    num_0_5 = 0
    num_1 = 0
    num_1_5 = 0
    num_2 = 0

    average_mean_sp = 0
    average_var_sp = 0
    average_cdf_sp = 0
    average_mean_rp = 0
    average_var_rp = 0
    average_cdf_rp = 0

    while True:
        req_idx += 1
        onid, dnid = load_trip(req_idx)
        d = round(get_the_minimum_duration_path_length(NYC_NET, onid, dnid) * 1.2, 2)
        shortest_path = get_the_minimum_duration_path(NYC_NET, onid, dnid)
        best_path = stochastic_shortest_path(d, onid, dnid)
        m_shortest, v_shortest = get_path_mean_and_var(shortest_path)
        m_best, v_best = get_path_mean_and_var(best_path)
        cdf_shortest = normal_cdf(d, shortest_path)
        cdf_best = normal_cdf(d, best_path)

        average_mean_sp += m_shortest
        average_var_sp += v_shortest
        average_cdf_sp += cdf_shortest
        average_mean_rp += m_best
        average_var_rp += v_best
        average_cdf_rp += cdf_best

        if shortest_path != best_path:
            num_different += 1
            difference = (cdf_best - cdf_shortest) * 100
            if difference > 2:
                num_2 += 1
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
            elif difference > 1.5:
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
            elif difference > 1:
                num_1 += 1
                num_0_5 += 1
            elif difference > 0.5:
                num_0_5 += 1
            print('req_index', req_idx, 'd', d, 'cdf_shortest', cdf_shortest, 'cdf_best', cdf_best)
            print('   average_mean (sp, rp):', round(average_mean_sp / req_idx, 2), round(average_mean_rp / req_idx, 2))
            print('   average_var (sp, rp):', round(average_var_sp / req_idx, 2), round(average_var_rp / req_idx, 2))
            print('   average_cdf (sp, rp):', round(average_cdf_sp / req_idx, 2), round(average_cdf_rp / req_idx, 2))
            print('   m_shortest, v_shortest:', (m_shortest, v_shortest), 'm_best, v_best:', (m_best, v_best))
            print('   different fraction:', round(num_different/req_idx, 3), round(num_0_5/req_idx, 3),
                  round(num_1/req_idx, 3), round(num_1_5/req_idx, 3), round(num_2/req_idx, 3))


def normal_cdf(d, path):
    mean, var = get_path_mean_and_var(path)
    return round(st.norm(mean, var).cdf(d), 4)


# find out the approximation quality, defined as the difference between the optimal solution (SSP) and ASSP
def verify_ssp_assp():
    K = [0.2667, 0.3333, 0.4167, 0.5208, 0.651, 0.8138, 1.0173, 1.2716, 1.5895, 1.9868, 2.4835, 3.1044, 3.8805,
         4.8506, 6.0633, 7.5791, 9.4739, 11.8424, 14.803, 18.5037, 23.1296, 28.9121, 36.1401, 45.1751, 56.4689,
         70.5861, 88.2326, 110.2907, 137.8634, 172.3293, 215.4116, 269.2645, 336.5807, 420.7258, 525.9073, 564.0]
    # k = [0.2, 0.3155, 0.4976, 0.7849, 1.2381, 1.9529, 3.0803, 4.8588, 7.664, 12.0888, 19.0683, 30.0774, 47.4425,
    #      74.8335, 118.0386, 186.1882, 293.684, 463.2426, 705]
    req_idx = 0
    num_different = 0

    test_idx = list(range(10000, 100000))
    for req_idx in tqdm(test_idx, desc='req_idx'):
    # while True:
        req_idx += 1
        onid, dnid = load_trip(req_idx)
        d = round(get_the_minimum_duration_path_length(NYC_NET, onid, dnid) * 1.2, 2)
        path1 = stochastic_shortest_path(d, onid, dnid)
        path2 = approximated_stochastic_shortest_path(K, d, onid, dnid)
        if path1 != path2:
            num_different += 1
            m1, v1 = get_path_mean_and_var(path1)
            phi1 = get_path_phi(d, m1, v1)
            m2, v2 = get_path_mean_and_var(path2)
            phi2 = get_path_phi(d, m2, v2)
            print('req_index', req_idx, 'd', d, 'path1', (m1, v1, phi1), 'path2', (m2, v2, phi2))


if __name__ == '__main__':
    start_time = time.time()

    find_out_how_many_trips_will_have_a_differnt_path()
    # verify_ssp_assp()

    print('...running time : %.05f seconds' % (time.time() - start_time))
