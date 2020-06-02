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

from ssp import get_the_minimum_duration_path, get_the_minimum_duration_path_length, stochastic_shortest_path

with open('NYC_NET.pickle', 'rb') as f:
    NYC_NET = pickle.load(f)
with open('./data/NYC_REQ_DATA_20160501.pickle', 'rb') as f:
    REQ_DATA = pickle.load(f)

if __name__ == '__main__':
    onid = 12
    dnid = 3788
    d = get_the_minimum_duration_path_length(NYC_NET, onid, dnid) * 1.2

    start_time = time.time()

    best_path = stochastic_shortest_path(d, onid, dnid)

    print('best_path', best_path)

    print('...running time : %.05f seconds' % (time.time() - start_time))
