"""
data file
"""

import pickle


with open('NYC_NET.pickle', 'rb') as f:
    NYC_NET = pickle.load(f)
with open('NYC_NOD_LOC.pickle', 'rb') as f:
    NOD_LOC = pickle.load(f)
with open('NYC_REQ_DATA_20160501.pickle', 'rb') as f:
    REQ_DATA = pickle.load(f)

