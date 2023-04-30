import h5py
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import re
import yaml

from collections.abc import MutableMapping

##############################################################
def load_yaml(config_yaml_path: str):

    with open(config_yaml_path) as f:
        data = yaml.load(f, Loader=yaml.BaseLoader)

    return data

##############################################################
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
    data = pd.read_csv(path, header=None, names=['Statistics'])['Statistics'].values.tolist()
    # remove unwanted tabs
    return  data 


##############################################################
def load_ims(ims_file_path: str) -> h5py.File:
    """
    loads a imaris file with .ims extension using h5py 

    Args:
        ims_file_path (str): path to .ims file 

    Returns:
        h5py.File: returns the data contained within the .ims file
    """
    return h5py.File(ims_file_path, 'r+') 


##############################################################
def get_object_names(full_data_file: h5py.File, search_for: str) -> list:
    """
    extracts the object names that we are interested in.

    Args:
        full_data_file (h5py.File): full imaris file in h5py File format
        search_for (str): string containing full or partial filename to search for

    Returns:
        list: a list of all the object names that match search_for parameter
    """

    values = full_data_file.get('Scene').get('Content').keys()
    storage = list()
    for item in values:
        if len(re.findall(search_for, item)):
            storage.append(item)
    return storage


##############################################################
def get_statistics_names(full_data_file: h5py.File, object_name: str) -> dict:
    """
    for a given object_name, extracts the statistics names and ids into a dict
    ex: statistics name = mean intensity, associated id=404

    Args:
        full_data_file (h5py._hl.files.File): full imaris file in h5py File format
        object_name (str): name of the object to get statistic names from

    Returns:
        dict: a dict where the keys=unique stats ID, value=static name
    """

    # get object specific data
    obj_specific_data = full_data_file['Scene8']['Content'][object_name]
    
    # rearrange data
    statistics_name = np.asarray(obj_specific_data['StatisticsType'])
    statistics_name = pd.DataFrame(statistics_name)
    
    # extract statistics names
    stats_name = statistics_name['Name']
    
    # extract statistics ID names
    stats_type = statistics_name['ID']
    
    # combine stats type and stats names
    return  dict(zip(stats_type, stats_name))


##################################################################
def get_stats_values(full_data_file: h5py.File, object_name: str) -> pd.DataFrame:
    """
    for a given object_name, extracts the statistics values for all object ids
    within the object

    Args:
        full_data_file (h5py._hl.files.File): full imaris file in h5py File format
        object_name (str): name of the object to get statistic names from

    Returns:
        pd.DataFrame: a pandas data frame that contains information about each object id
        where each object id has a stats id and associated stats value.
    """
    obj_specific_stats = full_data_file.get('Scene8').get('Content')[object_name]['StatisticsValue']
    obj_specific_stats = np.asarray(obj_specific_stats)
    return pd.DataFrame(obj_specific_stats)


##################################################################
def invert_stats_dict(stats_dict: dict) -> dict:
    """
    inverts a given dictionary from key: value to value: key
    additionally reorganizes statitics id values 

    Args:
        stats_dict (dict): dictionary containing statistics names as 
        keys and ids as values. This dict is the direct output of the 
        get_statistics_names() function. 

    Returns:
        dict : inverted and modified dict. 
    """
    
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


##################################################################
def flatten(input_dict: dict, parent_key='', sep='_') -> dict:
    """
    flattens a nested dictionary

    Args:
        input_dict (dict): _description_
        parent_key (str, optional): _description_. Defaults to ''.
        sep (str, optional): _description_. Defaults to '_'.

    Returns:
        dict: flattened dictionary
    """
    
    items = []
    for k, v in input_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


##################################################################
def create_del_list(inverted_stats_names: dict, valid_categories: list):
    """
    use the categories reqested by user to create a list of numbers we dont want
    if the use does not want any categories removed ie: len(valid_categories) == len(stats)
        we should skip this step
        
    [note]: user can request categories that are not in current files statistics
    Args:
        inverted_stats_names (dict): format -> key=numeric, value=name
        valid_categories (list): names
    """
    # map what the user wants to numeric values
    user_request_numeric = []
    
    # loop and update the values we want to delete
    for index, valid_category in enumerate(valid_categories):
        try:
            user_request_numeric.append(inverted_stats_names[valid_category])
        except KeyError:
            print(f'--- [warning] -- user requested category {valid_category} is NOT in current file')
            pass
            
    # take a local copy of the stats dict
    local_stats_copy = {v: k for k,v in inverted_stats_names.items()}

    # remove all numeric stats values we dont want
    for value in user_request_numeric:
        local_stats_copy.pop(value)
    
    # create a list of just the numeric values w/o the names
    del_list = list(local_stats_copy.keys())
    
    return del_list

##################################################################
def generate_csv(data_dict: dict, 
                 del_list: list, 
                 stats_names: dict, 
                 categories_list: list) -> pd.DataFrame:
    """
    Function used to clean up and generate final pd.DataFrame
    
    Args:
        data_dict (dict): dictionary containing all the objects and stats values
        del_list (list): list of stats_names (in integer form) that we do not want
        stats_names (dict): the final form of the stats names dictionary
        categories_list (list): list of categories user requires in final csv, in the 
                                order they want it in. 
    Returns:
        pd.DataFrame: final organized dataframe ready to be saved as a csv
    """
    
    # create the dataframe from the data dictionary
    # transpose such that the columns are the stats names (in integer form)
    dataframe = pd.DataFrame(data_dict).transpose()
    
    # drop the stats names we do not want
    dataframe = dataframe.drop(labels=del_list, axis=1)
    
    # map the integer values in the data frame to corresponding names
    new_column_names = {key: stats_names[key] for key in dataframe.columns}
        
    # rename columns
    dataframe = dataframe.rename(new_column_names, axis=1)
    
    # create id column
    dataframe['ID'] = dataframe.index

    # reorder columns to match user request
    dataframe = dataframe[categories_list]
    
    return dataframe
    
    
    
    
    
    
    
    
    
    
    
    
    
    