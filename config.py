"""
data file
"""

import pickle


with open('./graph/NYC_NET.pickle', 'rb') as f:
    NETWORK = pickle.load(f)
with open('./graph/NYC_REQ_DATA_20150505.pickle', 'rb') as f:
    REQ_DATA = pickle.load(f)

DEADLINE_COE = 1.2
