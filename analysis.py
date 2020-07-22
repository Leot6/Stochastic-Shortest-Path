"""
try to find out the percentage of how many trips would have a different path than the shortest path
"""

import time
import pickle
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import NETWORK, REQ_DATA, DEADLINE_COE
from stochastic_shortest_path import get_the_minimum_duration_path, get_the_minimum_duration_path_length, \
    stochastic_shortest_path, get_path_mean_and_var, get_path_phi
from approximated_ssp_query import approximated_stochastic_shortest_path

with open('./graph/NYC_NET.pickle', 'rb') as f:
    NETWORK = pickle.load(f)
# # parameters for Manhattan map
# map width and height (km)
MAP_WIDTH = 10.71
MAP_HEIGHT = 20.85
# coordinates
# (Olng, Olat) lower left corner
Olng = -74.0300
Olat = 40.6950
# (Olng, Olat) upper right corner
Dlng = -73.9030
Dlat = 40.8825


def find_out_how_many_trips_will_have_a_differnt_path():
    req_dur = []
    different_trips = []

    num_different = 0
    num_0_5 = 0
    num_1 = 0
    num_1_5 = 0
    num_2 = 0
    num_2_5 = 0
    num_3 = 0
    num_3_5 = 0
    num_4 = 0

    average_mean_0 = 0
    average_mean_0_5 = 0
    average_mean_1 = 0
    average_mean_1_5 = 0
    average_mean_2 = 0
    average_mean_2_5 = 0
    average_mean_3 = 0
    average_mean_3_5 = 0
    average_mean_4 = 0
    average_var_0 = 0
    average_var_0_5 = 0
    average_var_1 = 0
    average_var_1_5 = 0
    average_var_2 = 0
    average_var_2_5 = 0
    average_var_3 = 0
    average_var_3_5 = 0
    average_var_4 = 0

    average_mean_sp = 0
    average_var_sp = 0
    average_cdf_sp = 0
    average_mean_rp = 0
    average_var_rp = 0
    average_cdf_rp = 0

    for req_idx in tqdm(range(REQ_DATA.shape[0] - 1), desc='req_idx'):
        req_idx += 1
        onid = REQ_DATA.iloc[req_idx]['onid']
        dnid = REQ_DATA.iloc[req_idx]['dnid']
        ddl = get_the_minimum_duration_path_length(NETWORK, onid, dnid) * DEADLINE_COE
        if ddl == 0:
            continue
        shortest_path = get_the_minimum_duration_path(NETWORK, onid, dnid)
        best_path = stochastic_shortest_path(ddl, onid, dnid)
        m_shortest, v_shortest = get_path_mean_and_var(shortest_path)
        m_best, v_best = get_path_mean_and_var(best_path)
        cdf_shortest = normal_cdf(ddl, shortest_path)
        cdf_best = normal_cdf(ddl, best_path)

        average_mean_sp += m_shortest
        average_var_sp += v_shortest
        average_cdf_sp += cdf_shortest
        average_mean_rp += m_best
        average_var_rp += v_best
        average_cdf_rp += cdf_best

        req_dur.append(m_shortest)

        if shortest_path != best_path:
            num_different += 1
            difference = round((cdf_best - cdf_shortest) * 100, 2)
            different_trips.append((req_idx + 1, m_shortest, v_shortest, shortest_path,
                                    difference, m_best, v_best, best_path))
            if difference > 4:
                num_4 += 1
                num_3_5 += 1
                num_3 += 1
                num_2_5 += 1
                num_2 += 1
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
                average_mean_4 += m_shortest
                average_var_4 += v_shortest
            elif difference > 3.5:
                num_3_5 += 1
                num_3 += 1
                num_2_5 += 1
                num_2 += 1
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
                average_mean_3_5 += m_shortest
                average_var_3_5 += v_shortest
            elif difference > 3:
                num_3 += 1
                num_2_5 += 1
                num_2 += 1
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
                average_mean_3 += m_shortest
                average_var_3 += v_shortest
            elif difference > 2.5:
                num_2_5 += 1
                num_2 += 1
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
                average_mean_2_5 += m_shortest
                average_var_2_5 += v_shortest
            elif difference > 2:
                num_2 += 1
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
                average_mean_2 += m_shortest
                average_var_2 += v_shortest
            elif difference > 1.5:
                num_1_5 += 1
                num_1 += 1
                num_0_5 += 1
                average_mean_1_5 += m_shortest
                average_var_1_5 += v_shortest
            elif difference > 1:
                num_1 += 1
                num_0_5 += 1
                average_mean_1 += m_shortest
                average_var_1 += v_shortest
            elif difference > 0.5:
                num_0_5 += 1
                average_mean_0_5 += m_shortest
                average_var_0_5 += v_shortest
            else:
                average_mean_0 += m_shortest
                average_var_0 += v_shortest

            print('req_index', req_idx, 'd', ddl, 'cdf_shortest', cdf_shortest, 'cdf_best', cdf_best)
            print('   average_mean (sp, rp):', round(average_mean_sp / req_idx, 2), round(average_mean_rp / req_idx, 2))
            print('   average_var (sp, rp):', round(average_var_sp / req_idx, 2), round(average_var_rp / req_idx, 2))
            print('   average_cdf (sp, rp):', round(average_cdf_sp / req_idx, 2), round(average_cdf_rp / req_idx, 2))
            print('   m_shortest, v_shortest:', (m_shortest, v_shortest), 'm_best, v_best:', (m_best, v_best))
            print('   different fraction:', round(num_different / req_idx, 3), round(num_0_5 / req_idx, 3),
                  round(num_1 / req_idx, 3), round(num_1_5 / req_idx, 3), round(num_2 / req_idx, 3),
                  round(num_2_5 / req_idx, 3), round(num_3 / req_idx, 3), round(num_3_5 / req_idx, 3),
                  round(num_4 / req_idx, 3))
            if req_idx > 500:
                print('   num in different fraction:', num_different - num_0_5, num_0_5 - num_1, num_1 - num_1_5,
                      num_1_5 - num_2, num_2 - num_2_5, num_2_5 - num_3, num_3 - num_3_5, num_3_5 - num_4, num_4)
                print('   mean in different fraction:', round(average_mean_0 / (num_different - num_0_5), 2),
                      round(average_mean_0_5 / (num_0_5 - num_1), 2), round(average_mean_1 / (num_1 - num_1_5), 2),
                      round(average_mean_1_5 / (num_1_5 - num_2), 2), round(average_mean_2 / (num_2 - num_2_5), 2),
                      round(average_mean_2_5 / (num_2_5 - num_3), 2), round(average_mean_3 / (num_3 - num_3_5), 2),
                      round(average_mean_3_5 / (num_3_5 - num_4), 2), round(average_mean_4 / num_4, 2))
                print('   var in different fraction:', round(average_var_0 / (num_different - num_0_5), 2),
                      round(average_var_0_5 / (num_0_5 - num_1), 2), round(average_var_1 / (num_1 - num_1_5), 2),
                      round(average_var_1_5 / (num_1_5 - num_2), 2), round(average_var_2 / (num_2 - num_2_5), 2),
                      round(average_var_2_5 / (num_2_5 - num_3), 2), round(average_var_3 / (num_3 - num_3_5), 2),
                      round(average_var_3_5 / (num_3_5 - num_4), 2), round(average_var_4 / num_4, 2))

        if req_idx % 10000 == 0:
            with open('req_dur.pickle', 'wb') as f:
                pickle.dump(req_dur, f)
            with open('different_trips.pickle', 'wb') as f:
                pickle.dump(different_trips, f)

    with open('./analysis/req_dur.pickle', 'wb') as f:
        pickle.dump(req_dur, f)
    with open('./analysis/different_trips.pickle', 'wb') as f:
        pickle.dump(different_trips, f)


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
        req_idx += 1
        onid = REQ_DATA.iloc[req_idx]['onid']
        dnid = REQ_DATA.iloc[req_idx]['dnid']
        d = round(get_the_minimum_duration_path_length(NETWORK, onid, dnid) * DEADLINE_COE, 2)
        path1 = stochastic_shortest_path(d, onid, dnid)
        path2 = approximated_stochastic_shortest_path(K, d, onid, dnid)
        if path1 != path2:
            num_different += 1
            m1, v1 = get_path_mean_and_var(path1)
            phi1 = get_path_phi(d, m1, v1)
            m2, v2 = get_path_mean_and_var(path2)
            phi2 = get_path_phi(d, m2, v2)
            print('req_index', req_idx, 'd', d, 'path1', (m1, v1, phi1), 'path2', (m2, v2, phi2))


# find out the distribution of stochastic paths with different lengths
def time_analysis():
    col_req_id = 'req_id'
    col_mean_shortest = 'mean_shortest'
    col_var_shortest = 'var_shortest'
    col_path_shortest = 'path_shortest'
    col_improvement = 'improvement_on_arrival_prob(%)'
    col_mean_best = 'mean_best'
    col_var_best = 'var_best'
    col_path_best = 'path_best'
    # the travel times of all trips
    with open('./analysis/req_dur.pickle', 'rb') as f:
        req_dur = pickle.load(f)
    df1 = pd.DataFrame(req_dur, columns=[col_mean_shortest])
    # the set of trips that find a higher probability path than the minimum duration one
    with open('./analysis/different_trips.pickle', 'rb') as f:
        different_trips = pickle.load(f)
    df2 = pd.DataFrame(different_trips, columns=[col_req_id, col_mean_shortest, col_var_shortest, col_path_shortest,
                                                 col_improvement, col_mean_best, col_var_best, col_path_best])

    # bin travel time values into 50 sec intervals
    l_bound = int(min(req_dur) / 100) * 100
    u_bound = int(max(req_dur) / 100) * 100 + 50
    time_bins = list(range(l_bound, u_bound, 50))
    df1['time_bin'] = pd.cut(df1[col_mean_shortest], time_bins)
    df2['time_bin'] = pd.cut(df2[col_mean_shortest], time_bins)

    # number of trips in different time bins - all trips
    agg1_col = f'all trips - mean: {np.mean(req_dur):.2f} s, median: {np.median(req_dur):.2f} s'
    agg1 = df1.groupby(by=['time_bin'])[col_mean_shortest].agg(['count']).rename(columns={'count': agg1_col})
    # number of trips in different time bins - trips finding better paths
    agg2_col = f'trips finding better paths - mean: {df2[col_mean_shortest].mean():.2f} s, ' \
        f'median: {df2[col_mean_shortest].median():.2f} s'
    agg2 = df2.groupby(by=['time_bin'])[col_mean_shortest].agg(['count']).rename(columns={'count': agg2_col})

    # proportion of special trips in all trips in different time bins
    agg_2_divided_by_1_col = 'proportion of special trips in all trips'
    agg_2_divided_by_1 = pd.concat([agg1, agg2], axis=1).apply(lambda x: x[agg2_col]/x[agg1_col]*100, axis=1)
    agg_2_divided_by_1 = pd.DataFrame(agg_2_divided_by_1, columns=[agg_2_divided_by_1_col])
    # average improvement on arrival probability in different time bins
    agg2_prob_imp_col = 'average improvement on arrival probability'
    agg2_prob_imp = df2.groupby(by=['time_bin'])[col_improvement].agg(['mean']).\
        rename(columns={'mean': agg2_prob_imp_col})

    df3 = pd.concat([agg1, agg2, agg_2_divided_by_1, agg2_prob_imp], axis=1)
    ax_num = df3[[agg1_col, agg2_col]].plot(kind='bar', figsize=(16, 6))
    ax_num.set_xlabel('time bins (s)')
    ax_num.set_ylabel('number')
    ax_num.legend(loc=9)
    ax_pct = ax_num.twinx()
    ax_pct.set_ylabel('percentage (%)')
    df3[[agg_2_divided_by_1_col, agg2_prob_imp_col]].plot(ax=ax_pct)
    plt.tight_layout()
    plt.savefig('./analysis/analysis-for-different-times.png', dpi=300)
    plt.show()


# find out the distribution of stochastic paths with geography
def geo_analysis():
    col_req_id = 'req_id'
    col_mean_shortest = 'mean_shortest'
    col_var_shortest = 'var_shortest'
    col_path_shortest = 'path_shortest'
    col_improvement = 'improvement_on_arrival_prob(%)'
    col_mean_best = 'mean_best'
    col_var_best = 'var_best'
    col_path_best = 'path_best'
    # the set of trips that find a higher probability path than the minimum duration one
    with open('./analysis/different_trips.pickle', 'rb') as f:
        different_trips = pickle.load(f)
    df = pd.DataFrame(different_trips, columns=[col_req_id, col_mean_shortest, col_var_shortest, col_path_shortest,
                                                col_improvement, col_mean_best, col_var_best, col_path_best])
    shortest_paths = list(df.loc[:, col_path_shortest])
    best_paths = list(df.loc[:, col_path_best])
    idx = 5
    paths = [shortest_paths[idx], best_paths[idx]]
    plot_path(paths)


def plot_path(paths):
    fig = plt.figure(figsize=(MAP_WIDTH, MAP_HEIGHT))
    plt.xlim((Olng, Dlng))
    plt.ylim((Olat, Dlat))
    img = mpimg.imread('./graph/map.png')
    plt.imshow(img, extent=[Olng, Dlng, Olat, Dlat], aspect=(Dlng - Olng) / (Dlat - Olat) * MAP_HEIGHT / MAP_WIDTH)
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00)
    for path in tqdm(paths, desc='plotting path'):
        x = []
        y = []
        for node in path:
            [lng, lat] = NETWORK.nodes[node]['pos']
            x.append(lng)
            y.append(lat)
            plt.plot(x, y, '--')

    plt.savefig('./analysis/plotted_graph.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    start_time = time.time()

    # find_out_how_many_trips_will_have_a_differnt_path()
    # verify_ssp_assp()
    # time_analysis()
    geo_analysis()

    print('...running time : %.05f seconds' % (time.time() - start_time))
