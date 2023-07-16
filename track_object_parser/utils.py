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
import yaml



#-------------------------------------------------------------------------------------------------
# function to load yaml
def load_yaml(config_yaml_path: str):

    with open(config_yaml_path) as f:
        data = yaml.load(f, Loader=yaml.BaseLoader)

    return data

#-------------------------------------------------------------------------------------------------
def dict_to_yaml(data_dict: dict, save_path: str):
    
    with open(save_path, 'w') as f:
        yaml.dump(data_dict, f)

#-------------------------------------------------------------------------------------------------
# function to load data
def load_data(ims_file_path: str):
    '''
    args:
        file path to load with h5py
    '''
    return h5py.File(ims_file_path, 'r+')

#-------------------------------------------------------------------------------------------------
# get file categories list
# def get_categories(category_text_file_path: str) -> list:
    
#     lines =  pd.read_fwf("stats_categories.txt", delim_whitespace=True, header=None)[0].tolist()
#     lines = [x.strip('\t') for x in lines]
#     lines = [x.strip('\n') for x in lines]
    
#     return lines
def read_txt(path: str) -> list:
    '''
    args:
        path [string] -- path to the .txt file containing all the statistics
    function:
        reads and returns a list of the statistic names
    return:
        list
    '''
    # get a pandas dataframe of the statistics required
    data = pd.read_csv(path, sep=" ", header=None, names=['Statistics'])['Statistics'].values.tolist()
    # remove unwanted tabs
    return [x.strip('\t') for x in data]

#-------------------------------------------------------------------------------------------------
# function to get point names 
def get_points(full_data_file):
    '''
    args:
        num_points: number of points in the file
    returns:
        returns all the 'Points#' as a list
    '''
    values = full_data_file.get('Scene').get('Content').keys()
    storage = list()
    for item in values:
        if len(re.findall('Points', item)):
            storage.append(item)
    return storage

#-------------------------------------------------------------------------------------------------
def get_statistics_dict(dataframe, point):
    '''
    args:
        dataframe -- original imaris file in hd5py format
        point -- string to choose which point to use as the 'key'
    function:
        creates a dictionary that maps the numeric id to the statics type name
    return:
        the mapped dictionary
    interactions:
        none atm
    status: 
        done
    '''
    
    point_data = dataframe['Scene8']['Content'][point]
    
    statistics_name = np.asarray(point_data['StatisticsType'])
    statistics_name = pd.DataFrame(statistics_name)
    stats_name = statistics_name['Name']
    stats_type = statistics_name['ID']
    
    return  dict(zip(stats_type, stats_name))

#-------------------------------------------------------------------------------------------------
def get_stats(dataframe, point, category_type):
    '''
    args:
        dataframe -- full dataframe
        point -- ex: 'Point 0'
        category_type: the category of the statictics
    '''
    data = dataframe.get('Scene8').get('Content')[point][category_type]
    data = np.asarray(data)
    return pd.DataFrame(data)

#-------------------------------------------------------------------------------------------------
# create a function that groups all the information into one file
@njit
def get_object_id_info(track_id_names, track_start, track_end, object_id_info=None):
    '''
    args:
        track_id_data -- data from the 'Track0' file and the ID section in numpy format
        track_object_data -- data from the 'TrackObject0' file and the 'ID_Object' Section
        stats_value -- data from the 'StatisticsValue' section
    '''
    # get the max differnece between start and end
    difference = np.subtract(track_end, track_start)
        
    # create a zeros vector where index_zero = num_items, followed by the ids in each column
    # each row [track_id_name, start, end, count, info]
    #data = np.zeros((track_id_names.shape[0], difference.max() + 4), dtype=np.int64)  # original
    data = np.ones ((track_id_names.shape[0], difference.max() + 4), dtype=np.int64) * -1
    
    # put the extra info in first
    data[:, 0] = track_id_names
    data[:, 1] = track_start
    data[:, 2] = track_end
    data[:, 3] = difference # ie number of objects in that track
    
    # populate matrix
    for idx in range(data.shape[0]):
        data[idx, 4:4+data[idx, 3]] = np.reshape(object_id_info[data[idx,1]:data[idx, 2]], -1)
        
    return data

#-------------------------------------------------------------------------------------------------
# create dict
def create_dict(track_and_object_id_info):
    '''
    creates a dictionary so we can insert all the stats values for each object id which is the key
    '''
    storage = dict()
    
    for i in range(track_and_object_id_info.shape[0]):
        count_i = track_and_object_id_info[i, 3]
        for j in track_and_object_id_info[i, 4:4+count_i]:
            storage[j] = {}
            
    return storage

#-------------------------------------------------------------------------------------------------
# function to extract the data
def extract_data(object_info_matrix, stat_values):
    '''
    extacts and puts the stats value data into the dict from create_dict
    '''
    # create the dict to store the objct information in
    data_dict = create_dict(object_info_matrix)
    
    # data array
    data_array = np.array(stats_values, dtype=np.float32)[:, 1:]
    
    # loop and fill 
    for i in range(data_array.shape[0]):
        try:
            #data_dict[data_array[i, 0].astype(np.int64)].append({data_array[i, 1].astype(np.int64): data_array[i, 2]})
            data_dict[data_array[i, 0].astype(np.int64)][data_array[i, 1].astype(np.int64)] = data_array[i, 2]
        except KeyError:
            pass
        
    return data_dict

#-------------------------------------------------------------------------------------------------
def convert_to_matrix(track_id_data, track_object_data):
    '''
    args:
        track_id_data: dataframe with all the track_id information specifying which object_id indices belong to which track_id
        track_object_data: dataframe specifying which object ids are at which indices
    '''
    # get the track and object id information in one np array
    names = np.array(track_id_data['ID'], dtype=np.int64)
    start = np.array(track_id_data['IndexTrackObjectBegin'], dtype=np.int64)
    end = np.array(track_id_data['IndexTrackObjectEnd'], dtype=np.int64)
    object_id_info = np.array(track_object_data, dtype=np.int64)

    # create a single matrix 
    track_and_object_id_info = get_object_id_info(names, start, end, object_id_info)
    
    return track_and_object_id_info

#-------------------------------------------------------------------------------------------------
# function to extract the data
def extract_data(object_info_matrix, stats_values):
    '''
    extacts and puts the stats value data into the dict from create_dict
    given the matrix that has all the track_ids and the object ids
    and the stats_values that has all the information of each statistics value along with 
    the object id it belongs to, create a single dictionary that puts all the information 
    for each object_id and each track_id into a easily searchable dictionary. 
    '''
    # create the dict to store the objct information in: objkect_ids are keys
    data_dict = create_dict(object_info_matrix)
    
    # update the data dict to also include the track_ids as keys
    for idx in object_info_matrix[:, 0]:
        data_dict[idx] = {}
    
    # convert the dataframe into an array data array
    data_array = np.array(stats_values, dtype=np.float64)[:, 1:]
    
    # loop and fill 
    for i in range(data_array.shape[0]):
        try:
            #data_dict[data_array[i, 0].astype(np.int64)].append({data_array[i, 1].astype(np.int64): data_array[i, 2]})
            data_dict[data_array[i, 0].astype(np.int64)][data_array[i, 1].astype(np.int64)] = data_array[i, 2]
        except KeyError:
            pass
        
    return data_dict

#-------------------------------------------------------------------------------------------------
def invert_stats_dict(stats_dict: dict=None):
    '''
    creates a inverted_stats_dict --> [Statistics Type: Numeric Value]
    '''
    
    # sort the dict
    sorted_dict_key = sorted(stats_dict.keys())
    stats_dict = {key: str(stats_dict[key]).strip('b')[1:-1] for key in sorted_dict_key }
    
    # create an empty dictionary
    storage = {}
    
    for key in stats_dict.keys():
        # if the word is not in the new storage dict
        if stats_dict[key] not in storage.keys():
            storage[stats_dict[key]] = key
        else:
            # get the value inside the key 
            current_value = storage[stats_dict[key]]
            
            # if its a single value create a dict else create a dict
            if type(current_value) != dict:
                # then its the first value
                storage[stats_dict[key]] = {}
                storage[stats_dict[key]]['channel_1'] = current_value
                
                # current value
                storage[stats_dict[key]]['channel_2'] = key
            else:
                # get the length of the dict
                count = len(current_value.keys())
                # updated count
                count += 1 
                # create new key
                new_key = f"channel_{count}"
                current_value[new_key] = key
                storage[stats_dict[key]] = current_value
                
    return storage

#-------------------------------------------------------------------------------------------------
# custom decorator for catching key errors if key error return np.NAN
def return_on_failure(func):
    def applicator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError:
            return np.nan
    return applicator
#-------------------------------------------------------------------------------------------------
def strip(categories: str, remove_list: list):
    '''
    function:
        strips the categories of unwanted characters and letters
    '''
    # create a dict mapping input to cleaned up ouput
    storage = {key: [] for key in categories}
    
    # loop over the categories and clean it up
    for category in categories:
        # use try/except in case remove values does not exist
        # create a list by splitting
        temp_list = category.split('_')
        # filter list
        for char in remove_list:
            try:
                temp_list.remove(char)
            except ValueError:
                pass
        # combine into a string
        cleaned_str = " ".join(temp_list)
        # update storage
        storage[category].append(cleaned_str)
        
    return storage
#-------------------------------------------------------------------------------------------------
def count_channels(statistics_dict: dict, stats_name: str='Intensity Mean') -> int:
    '''
    args:
        statistics_dict [dict] -- the statistics_dict from the get_statistics_dict function
    function:
        to return the number of channels in the file for given stats_name
        as a default we will use "Intensity Mean" as the default statistics name
    return:
        int 
    '''
    # convert the values of the dict into a string
    values_list = list(statistics_dict.values())
    # convert from byte to string
    values_list = [x.decode('utf-8') for x in values_list]
    
    return values_list.count(stats_name)
#-------------------------------------------------------------------------------------------------
def create_functions_dict(categories: list, remove_list: list, statistics_dict: dict) -> dict:
    '''
    args:
        categories [list] -- the stats categories list we are interested in
        remove_list [list] -- the key words to be removed from the input categories list
    function:
        create a dictionary that will map the stats categories into ...
    
    '''
    # dict containing special items 
    special_items = {
        'Channel_Name': 'create_channel_name_column',
        'ID': 'create_track_id_column',
        'ID_Object': 'create_object_id_column'
    }
    
    # create the conversion dict below
    conversion_dict = strip(categories, remove_list)
    
    # initialize an empty dict
    functions_dict = {}
    
    # create a list of keys to be deleted
    del_list = []
    
    # update the functions dict for each category that requires a channel
    for key in conversion_dict.keys():
        
        # get the count
        count = count_channels(statistics_dict, stats_name=conversion_dict[key][0])
        
        # count greater than 1 consider it a channel
        if count > 1:
            # modify existing key value pair
            new_key = key + f"_Image_1_Channel_1"
            functions_dict[new_key] = copy.deepcopy(conversion_dict[key])
            functions_dict[new_key].extend(['channel_1'])
            
            for idx in range(2, count+1):
                new_key = key + f"_Image_1_Channel_{idx}"
                functions_dict[new_key] = copy.deepcopy(conversion_dict[key])
                functions_dict[new_key].extend([f"channel_{idx}"])
                
            del_list.append(key)
        elif count == 1 or count == 0:
            if key not in list(special_items.keys()):
                functions_dict[key] = conversion_dict[key][0]
            else:
                functions_dict[key] = special_items[key]
        
        else:
            pass
            
    return functions_dict