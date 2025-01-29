# %%

import gzip
import pickle
import time
import numpy as np
import pandas as pd
import pickle

with open('results_section_A.pkl', 'rb') as f:
    data = pickle.load(f)