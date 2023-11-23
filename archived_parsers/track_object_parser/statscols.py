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
from stats import Stats


class StatsColumns:
    
    def __init__(
        self,
        channel_name,
        stats_dict,
        track_id_data,
        track_object_data,
        stats_values,
        track_and_object_id_info,
        stats_data,
        inv_stats_dict):
        
        self.channel_name = channel_name
        self.stats_dict = stats_dict
        self.track_id_data = track_id_data
        self.track_object_data = track_object_data
        self.stats_values = stats_values
        self.track_and_object_id_info = track_and_object_id_info
        self.stats_data = stats_data
        self.inv_stats_dict = inv_stats_dict
        self.obj_ids = self.create_object_id_column()
    
    # ----------------------------------------------------------------------------------------------------------------
    def create_channel_name_column(self):

        track_ids = np.ones((self.obj_ids.shape[0], 1), dtype=int)
        count_start = self.track_and_object_id_info[:, 1]
        count_end = self.track_and_object_id_info[:, 2]
        counts = np.dstack((count_start, count_end)).squeeze()

        # expansion 
        for idx, data in enumerate(counts):
            start, stop = data
            value = self.channel_name
            track_ids[start:stop] = track_ids[start:stop] * int(value)

        return track_ids
    
    # ----------------------------------------------------------------------------------------------------------------
    #@njit
    # CREATE A FUNCTION THAT WILL CREATE THE OBJECT ID COLUMN
    def create_object_id_column(self):
        obj_ids = self.track_and_object_id_info[:, 4:].ravel()
        #idx = np.nonzero(obj_ids)
        idx = np.argwhere(obj_ids > -1).astype(np.int64) 

        updated_obj_ids = obj_ids[idx]
        
        return updated_obj_ids.reshape(updated_obj_ids.shape[0], 1)
    
    # ----------------------------------------------------------------------------------------------------------------
    # CREATE A FUNCTION THAT WILL CREATE THE TRACK ID COLUMN
    def create_track_id_column(self):

        track_ids = np.ones((self.obj_ids.shape[0], 1))
        count_start = self.track_and_object_id_info[:, 1]
        count_end = self.track_and_object_id_info[:, 2]
        counts = np.dstack((count_start, count_end)).squeeze()

        # expansion 
        for idx, data in enumerate(counts):
            start, stop = data
            value = self.track_and_object_id_info[idx, 0]
            track_ids[start:stop] = track_ids[start:stop] * value

        return track_ids
    # ----------------------------------------------------------------------------------------------------------------
    def universal_create_track_channel_value_column(self, stats_name: str, channel_name: str):
        '''only the channel passed is extracted'''
        # storage
        channel_storage = np.zeros((self.track_and_object_id_info.shape[0], 1))

        # first get all the x,y,z positions for each object trackid
        for idx, track_id in enumerate(self.track_and_object_id_info[:, 0]):
            channel_storage[idx] = getattr(Stats, 'universal_channel_stats')(self.stats_data, self.inv_stats_dict, track_id, channel_name.lower(), stats_name)

        track_ids = np.ones((self.obj_ids.shape[0], 1))
        count_start = self.track_and_object_id_info[:, 1]
        count_end = self.track_and_object_id_info[:, 2]
        counts = np.dstack((count_start, count_end)).squeeze()

        for idx, data in enumerate(counts):
            start, stop = data
            values = channel_storage[idx]
            track_ids[start:stop] = np.multiply(track_ids[start:stop], values)

        return track_ids
    
    # ----------------------------------------------------------------------------------------------------------------
    def universal_create_stats_column(self, stats_name: str):
        '''
        args:
            stats_name [string] -- what we want here is the name of the statistic we want to pull out
            drop_duplicates [bool] -- bool value to determine whether or not to keep all the object ids
        function:
            this funciton simply calls the right function from stats.py file and gathers the statistics
            once gathered it will either redistribute the stats so that we can all the statistics organized 
            per object or simply gather the stats for the last row. 
        
        '''
        # create a array to store the statistics data into
        # this array has the same length as the number of track ids 
        track_statistics = np.zeros((self.track_and_object_id_info.shape[0]), dtype=np.float32)

        # pull the statistics data out
        # loop over each track id and get the statistics defined by stats_name
        for idx, track_id in enumerate(self.track_and_object_id_info[:, 0]):
            track_statistics[idx] = getattr(Stats, 'universal_stats')(self.stats_data, self.inv_stats_dict, track_id, stats_name)

        object_ids = np.ones((self.obj_ids.shape[0], 1)).astype(np.float32) # simple storage array to store all the object ids                             
        count_start = self.track_and_object_id_info[:, 1] # specifies where the object ids start
        count_end = self.track_and_object_id_info[:, 2] # specifies where the object ids end
        counts = np.dstack((count_start, count_end)).squeeze() 

        # we are going to expand the data into each track id since the values are the same for each object_id in a certian track
        for idx, data in enumerate(counts):
            start, stop = data
            values = track_statistics[idx] # get the track wise data
            object_ids[start:stop] = np.multiply(object_ids[start:stop], values) # expand the values into each object within that track

        # return the expanded object_ids matrix
        return object_ids
    
    
    