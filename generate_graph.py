"""
generate the graph of Manhattan using networkx
"""

import time
import pickle
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


# get the duration based on haversine formula
def get_haversine_distance(olng, olat, dlng, dlat):
    dist = (6371000 * 2 * math.pi / 360 * np.sqrt((math.cos((olat + dlat) * math.pi / 360)
                                                   * (olng - dlng)) ** 2 + (olat - dlat) ** 2))
    return dist


def load_Manhattan_graph():
    aa = time.time()
    print('Loading edges and nodes data...')
    edges = pd.read_csv('./graph/edges.csv')
    nodes = pd.read_csv('./graph/nodes.csv')
    travel_time_edges = pd.read_csv('./graph/time-on-week.csv', index_col=0)
    # consider the travels times on different hours as samples, and compute the sample mean and standard deviation
    mean_travel_times = travel_time_edges.mean(1)
    std_travel_times = travel_time_edges.std(1)
    G = nx.DiGraph()
    num_edges = edges.shape[0]
    rng = tqdm(edges.iterrows(), total=num_edges, ncols=100, desc='Generating Manhattan Graph...')
    for i, edge in rng:
        u = edge['source']
        v = edge['sink']
        u_pos = np.array([nodes.iloc[u - 1]['lng'], nodes.iloc[u - 1]['lat']])
        v_pos = np.array([nodes.iloc[v - 1]['lng'], nodes.iloc[v - 1]['lat']])
        G.add_node(u, pos=u_pos)
        G.add_node(v, pos=v_pos)

        mean_travel_time = round(mean_travel_times.iloc[i], 2)
        std = round(std_travel_times.iloc[i], 2)

        # artificial variance
        variance = round(std * 0.3, 2)
        # variance = round(mean_travel_time / 100 * random.randint(10, 90), 2)
        print('mean', mean_travel_time, 'std', std, 'var', variance)

        travel_dist = get_haversine_distance(u_pos[0], u_pos[1], v_pos[0], v_pos[1])
        G.add_edge(u, v, dur=mean_travel_time, var=variance, dist=travel_dist)

    # store_map_as_pickle_file
    with open('NYC_NET.pickle', 'wb') as f:
        pickle.dump(G, f)
    print('Saving the graph as a pickle file...')

    print('...running time : %.05f seconds' % (time.time() - aa))


if __name__ == '__main__':
    load_Manhattan_graph()

