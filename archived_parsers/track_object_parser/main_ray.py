import h5py
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
from utils import *
from stats import Stats
from statscols import StatsColumns
import yaml
import glob
import os
import ray
import time

def run(config: dict, data_path: str, categories: list, drop_duplicates: bool):
    '''
    args:
        data_path: path to imaris file
    '''
    # load the data
    full_data_file = load_data(data_path)

    # get the points info inside the file
    points = get_points(full_data_file)
    
    # storage to store multiple dataframes
    dataframe_storage = list()

    full_storage = {}
    
    # metadata storage
    metadata_storage = {}

    # loop over each point
    for idx, point in enumerate(points):

        # create a dictionary that maps the statistics name to the 
        stats_dict = get_statistics_dict(full_data_file, point)
        
        # create the functions dict
        remove_list = read_txt(config['remove_list_path'])
        functions_dict = create_functions_dict(categories, remove_list, stats_dict)

        try:
            # get the track information
            track_id_data = get_stats(full_data_file, point, 'Track0')

            # get the track object information
            track_object_data = get_stats(full_data_file, point, 'TrackObject0')

            # get the stistics value information
            stats_values = get_stats(full_data_file, point, 'StatisticsValue')

            # get the track and object id information in one np array
            track_and_object_id_info = convert_to_matrix(track_id_data, track_object_data)

            # create a dict to extract the data 
            stats_data = extract_data(track_and_object_id_info, stats_values)

            # invert the stats dict ie: swap key and values
            inv_stats_dict = invert_stats_dict(stats_dict)

            # initialize the class to create all the necessary columns
            statscols = StatsColumns(
                idx,
                stats_dict,
                track_id_data,
                track_object_data,
                stats_values,
                track_and_object_id_info,
                stats_data,
                inv_stats_dict)

            # get the number of object in current point
            num_points = statscols.obj_ids.shape[0]

            # create a empty storage dict to store the data from each point
            storage_dict = {}

            # update metadata
            metadata_storage[point] = {'num_obj_ids': num_points, 'num_track_ids': statscols.track_and_object_id_info.shape[0]}
            
            # grab the special items
            for key in functions_dict.keys():

                if type(functions_dict[key]) == list: 
                    storage_dict[key] = getattr(statscols, 'universal_create_track_channel_value_column')(*functions_dict[key])
                else:
                    if key not in config['special_items']:
                        storage_dict[key] = getattr(statscols, 'universal_create_stats_column')(functions_dict[key])
                    else:
                        storage_dict[key] = getattr(statscols, functions_dict[key])()
                        
            full_storage[point] = storage_dict

            # update dataframe
            points_data_arr = pd.DataFrame(
                data=np.hstack(list(full_storage[point].values())),
                columns=list(functions_dict.keys()))

            dataframe_storage.append(points_data_arr)

            #print(f'info: found track')
            
        except (KeyError, AttributeError):
            print(f'info -- no track')
            pass
        
    # concatenate all the points and return
    return pd.concat(dataframe_storage), metadata_storage


@ray.remote
def subprocess(config: dict, categories: list, drop_duplicates: bool, data_path: str, keep_id: bool):
    '''
    args:
        config [dict] --
        categories [list] --
        drop_duplicates [bool] --
        data_path [str] -- 
        keep_id [bool] --
    function:
        this function is used with ray, it runs the entire pipeline for a single imaris file
        it generates all the statistics for each point in the imaris file and saves it to a csv
    returns:
        None
    '''
    
    try:
        #print(f"\ninfo: data path -- {data_path}")
        
        # get the name of the imaris file
        imaris_name = os.path.basename(data_path).split('.')[0]
        # create the csv file name
        csv_name = f"{imaris_name}.csv"
        # create the metadata file name
        metadata_name = f"{imaris_name}.yaml"

        # generate data 
        data_frame, metadata = run(config, data_path, categories, drop_duplicates)

        # remove unwanted columns with NO/EMPTY values
        data_frame.dropna(how='all', axis=1, inplace=True)

        # save data_frame
        # create directory to store csv file
        save_path = os.path.join(config['save_dir'], config['data_dir'])

        # drop the duplicates and only keep info for a single track
        if drop_duplicates:
            # drop the duplicates and keep only the last row
            data_frame = data_frame.drop_duplicates(subset=['ID'], keep='last', inplace=False, ignore_index=True)
            
        # switch to indicate whether or not to drop track id information column
        if keep_id == False:
            data_frame = data_frame.drop('ID', axis=1)
            
        # finally save the data csv
        data_frame.to_csv(os.path.join(config['save_dir'], csv_name), index=False)
        
        # save the metadata yaml file
        dict_to_yaml(metadata, os.path.join(config['save_dir'], metadata_name))
        
    except (ValueError, AttributeError):
        print(f'info -- Skipping ""{data_path}"" File - No Tracks Found\n')
        pass
    
    
def main(config_path: str, drop_duplicates: bool=True, keep_id: bool=False) -> None:
    '''
    args:
        config_path: path to the config yaml file
    '''
    # load yaml file as a dictionary
    config = load_yaml(config_path)
    
    # get the statistics categories
    categories = read_txt(config['stats_category_path'])
    
    # create saving directory
    None if os.path.exists(config['save_dir']) else os.mkdir(config['save_dir'])
    
    # get all the imaris files in the directory
    data_paths = glob.glob(os.path.join(config['data_dir'], '*.ims'))
      
    # create an empty list to store all the subprocesses to be executed by ray
    processes = []
    
    # apped each function to be executed to the list
    print('info -- generating subprocesses')
    print(f'info -- subprocess being created for the following {len(data_paths)} imaris files')
    for idx, path in enumerate(data_paths):
        print(f'info -- file {idx} : {path}')
        processes.append(subprocess.remote(config, categories, drop_duplicates, path, keep_id))
        
    # run ray to lauch each function in a parallel manner
    print('info -- running subprocesses:')
    ray.get(processes)
    
    
    