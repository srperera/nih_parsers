import h5py
import io
import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from rich import print
import copy
import pandas as pd
import re
from tqdm import tqdm
import timeit
import re
from utils import return_on_failure


class Stats:
#-------------------------------------------------------------------------------------------------
    @staticmethod
    @return_on_failure
    def universal_channel_stats(stats_data, inv_stats_dict, track_id, channel, name):
        '''
        a function to pull data from images with channel wise stats
        '''
        return stats_data[track_id][inv_stats_dict[name][channel]]
#-------------------------------------------------------------------------------------------------
    @staticmethod
    @return_on_failure
    def universal_stats(stats_data, inv_stats_dict, track_id, name):
        '''
        a function to pull data from individual stats 
        '''
        return stats_data[track_id][inv_stats_dict[name]]
#-------------------------------------------------------------------------------------------------