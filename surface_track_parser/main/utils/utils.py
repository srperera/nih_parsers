import h5py
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import re
import yaml
import os

from collections.abc import MutableMapping
from typing import Dict, List


##############################################################
def load_yaml(config_yaml_path: str):
    with open(config_yaml_path) as f:
        data = yaml.load(f, Loader=yaml.BaseLoader)

    return data


##############################################################
def read_txt(path: str) -> List:
    """
    args:
        path [string] -- path to the .txt file containing all the statistics
    function:
        reads and returns a list of the statistic names
    return:
        list
    """
    # get a pandas dataframe of the statistics required
    data = pd.read_csv(path, header=None, names=["Statistics"])[
        "Statistics"
    ].values.tolist()
    # remove unwanted tabs
    return data


##############################################################
def load_ims(ims_file_path: str) -> h5py.File:
    """
    loads a imaris file with .ims extension using h5py

    Args:
        ims_file_path (str): path to .ims file

    Returns:
        h5py.File: returns the data contained within the .ims file
    """
    return h5py.File(ims_file_path, "r+")


##############################################################
def get_object_names(full_data_file: h5py.File, search_for: str) -> List:
    """
    extracts the object names that we are interested in.

    Args:
        full_data_file (h5py.File): full imaris file in h5py File format
        search_for (str): string containing full or partial filename to search for

    Returns:
        list: a list of all the object names that match search_for parameter
    """

    values = full_data_file.get("Scene8").get("Content").keys()
    storage = list()
    for item in values:
        if len(re.findall(search_for, item)):
            storage.append(item)
    return storage


##############################################################
def get_statistics_names(full_data_file: h5py.File, object_name: str) -> Dict:
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
    obj_specific_data = full_data_file["Scene8"]["Content"][object_name]

    # rearrange data
    statistics_name = np.asarray(obj_specific_data["StatisticsType"])
    statistics_name = pd.DataFrame(statistics_name)

    # extract statistics names
    stats_name = statistics_name["Name"]

    # extract statistics ID names
    stats_type = statistics_name["ID"]

    # combine stats type and stats names
    return dict(zip(stats_type, stats_name))


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
    # get the track ids
    obj_specific_stats = full_data_file.get("Scene8").get("Content")[object_name][
        "StatisticsValue"
    ]
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
    stats_dict = {key: str(stats_dict[key]).strip("b")[1:-1] for key in sorted_dict_key}

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
                storage[stats_dict[key]]["channel_1"] = current_value

                # current value
                storage[stats_dict[key]]["channel_2"] = key
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
def flatten(input_dict: dict, parent_key="", sep="_") -> Dict:
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
def create_del_list(inverted_stats_names: dict, valid_categories: List):
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
            # print(
            #     f"--- [warning] -- user requested category {valid_category} is NOT in current file"
            # )
            pass

    # take a local copy of the stats dict
    local_stats_copy = {v: k for k, v in inverted_stats_names.items()}

    # remove all numeric stats values we dont want
    for value in user_request_numeric:
        local_stats_copy.pop(value)

    # create a list of just the numeric values w/o the names
    del_list = list(local_stats_copy.keys())

    return del_list


##################################################################
def generate_csv(
    data_dict: dict, del_list: list, stats_names: dict, categories_list: list
) -> pd.DataFrame:
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

    # print(new_column_names)

    # rename columns
    dataframe = dataframe.rename(new_column_names, axis=1)

    # create id column
    dataframe["ID"] = dataframe.index

    # reorder columns to match user request
    dataframe = dataframe[categories_list]

    return dataframe


##################################################################
def check_anomalies(out_stats_names: dict, save_path: str) -> None:
    """
    checks to see if the numerical order of the stats parameters are correct.
    if the values are ascending there is no anomaly.
    if the values are not in direct ascending order there is an anomaly.
    Args:
        out_stats_names (dict): _description_
    """
    # step 1: invert the {num: stat_name} dict
    counts = {}

    for item, param in out_stats_names.items():
        if counts.get(param) == None:
            counts[param] = [item]
        else:
            counts[param].append(item)

    # step 2: check the difference between values to ensure ascending order
    out_of_order = []
    for param, values in counts.items():
        if len(values) > 1:
            # check to see if values are ascending
            if sorted(values) == values:
                pass
            else:
                out_of_order.append([param, "--", values])

    # pd.DataFrame(out_of_order, columns=['stat_name', 'numeric_item_code']).to_('anomaly_items.csv')

    if len(out_of_order) != 0:
        base_path = os.path.dirname(save_path)
        ims_name = os.path.basename(save_path).split(".")[0]
        fname = os.path.join(base_path, f"{ims_name}_anomalies.txt")
        np.savetxt(fname, np.array(out_of_order, dtype=object), fmt="%s")


##################################################################
def get_track_ids(full_data_file: h5py.File, object_name: str) -> pd.Series:
    """
    Gets all the track ids for a given surface
    Args:
        full_data_file (h5py.File): _description_
        object_name (str): _description_
    """
    track_ids = pd.DataFrame(
        np.asarray(full_data_file.get("Scene8").get("Content")[object_name]["Track0"])
    )
    return track_ids["ID"]


##################################################################
def filter_by_object_id(
    stats_values: pd.DataFrame, object_ids: pd.Series
) -> pd.DataFrame:
    """
    Filter a given stats values dictionary to only contain stats values for a given
    set of object ids.
    Args:
        stats_values (pd.DataFrame): _description_
        object_ids (pd.Series): _description_

    Returns:
        pd.DataFrame: _description_
    """
    filtered_df = stats_values[stats_values["ID_Object"].isin(values=object_ids)]

    return filtered_df


##################################################################
def filter_stats_dict(stats_dict: Dict, filtered_df: pd.DataFrame) -> Dict:
    """
    Stats Dict contains a key=numeraical stats value, and value = str r
    representation of the stats type. This is for every object in the .ims file.
    Not all objects contains all the stats, so we will use the filtered dataframe
    to identify the stats associated with the filtered df and filter the
    stats dict to only contain the data we need
    Args:
        stats_dict (Dict): _description_
        filtered_df (pd.DataFrame): _description_

    Returns:
        Dict: _description_
    """
    filtered_stats = {}
    for stat_type in filtered_df["ID_StatisticsType"].to_list():
        filtered_stats[stat_type] = stats_dict[stat_type]

    return filtered_stats


##################################################################
def get_first_timeframe_objects(
    stats_values: pd.DataFrame, stats_dict: Dict
) -> pd.Series:
    """
    Gets all the track ids for a given surface
    Args:
        full_data_file (h5py.File): _description_
        object_name (str): _description_
    """
    # reverse stats dict
    inv_stats_dict = invert_stats_dict(stats_dict)

    # get stats value for Time Index
    time_index = inv_stats_dict["Time Index"]

    # filter stats value based on time index == 1
    temp = stats_values[stats_values["ID_StatisticsType"] == time_index]
    first_objects = temp[temp["Value"] == 1]["ID_Object"]

    return first_objects


##################################################################
def available_categories_first(ims_file_path: str, valid_surface: int, save_path: str):
    # load the imaris file
    data = load_ims(ims_file_path)

    # get surface we want to parse
    surface_name = get_object_names(full_data_file=data, search_for="Surface")[
        valid_surface
    ]

    # get all the statistics names
    surface_stats_names = get_statistics_names(
        full_data_file=data, object_name=surface_name
    )

    # get the statistics values in the surface
    surface_stats_values = get_stats_values(
        full_data_file=data, object_name=surface_name
    )

    # get only the traack ids
    track_ids = get_first_timeframe_objects(
        stats_values=surface_stats_values, stats_dict=surface_stats_names
    )

    # filter full stats values based on track ids
    filtered_stats_df = filter_by_object_id(
        stats_values=surface_stats_values, object_ids=track_ids
    )

    # filter stats dict
    filtered_dict = filter_stats_dict(
        stats_dict=surface_stats_names, filtered_df=filtered_stats_df
    )

    # invert dictionary + name modifications
    # this step is a cosmetic step
    inverted_stats_names = invert_stats_dict(filtered_dict)
    inverted_stats_names = flatten(inverted_stats_names)

    # filepath
    save_path = os.path.join(save_path, "stats_categories_first.txt")
    np.savetxt(save_path, list(inverted_stats_names.keys()), fmt="%s")

    # print(f"Saved Stats Categories in directory: {os.getcwd()}")


##################################################################
def available_categories_track(ims_file_path: str, valid_surface: int, save_path: str):
    # load the imaris file
    data = load_ims(ims_file_path)

    # get surface we want to parse
    surface_name = get_object_names(full_data_file=data, search_for="Surface")[
        valid_surface
    ]

    # get all the statistics names
    surface_stats_names = get_statistics_names(
        full_data_file=data, object_name=surface_name
    )

    # get the statistics values in the surface
    surface_stats_values = get_stats_values(
        full_data_file=data, object_name=surface_name
    )

    # get only the traack ids
    track_ids = get_track_ids(full_data_file=data, object_name=surface_name)

    # filter full stats values based on track ids
    filtered_stats_df = filter_by_object_id(
        stats_values=surface_stats_values, object_ids=track_ids
    )

    # filter stats dict
    filtered_dict = filter_stats_dict(
        stats_dict=surface_stats_names, filtered_df=filtered_stats_df
    )

    # invert dictionary + name modifications
    # this step is a cosmetic step
    inverted_stats_names = invert_stats_dict(filtered_dict)
    inverted_stats_names = flatten(inverted_stats_names)

    # filepath
    save_path = os.path.join(save_path, "stats_categories_track.txt")
    np.savetxt(save_path, list(inverted_stats_names.keys()), fmt="%s")

    # print(f"Saved Stats Categories in directory: {os.getcwd()}")


##################################################################
def available_categories_surface(
    ims_file_path: str, valid_surface: int, save_path: str
):
    # load the imaris file
    data = load_ims(ims_file_path)

    # get surface we want to parse
    surface_name = get_object_names(full_data_file=data, search_for="Surface")[
        valid_surface
    ]

    # get all the statistics names
    surface_stats_names = get_statistics_names(
        full_data_file=data, object_name=surface_name
    )

    # invert dictionary + name modifications
    # this step is a cosmetic step
    inverted_stats_names = invert_stats_dict(surface_stats_names)
    inverted_stats_names = flatten(inverted_stats_names)

    # save path
    save_path = os.path.join(save_path, "stats_categories_surface.txt")
    np.savetxt(save_path, list(inverted_stats_names.keys()), fmt="%s")

    # print(f"Saved Stats Categories in directory: {os.getcwd()}")


##################################################################
def get_category_function(parser_type: str):
    if parser_type == "track":
        return available_categories_track
    elif parser_type == "surface":
        return available_categories_surface
    elif parser_type == "first":
        return available_categories_first
    else:
        raise ValueError("Invalid Parser Type")


# def get_save_filepath(parser_type: str, filename: str):
#     if parser_type == "track":
#         return os.path.splitext(filename)[0] + "_track.csv"
#     elif parser_type == "surface":
#         return os.path.splitext(filename)[0] + "_surface.csv"
#     elif parser_type == "first":
#         return os.path.splitext(filename)[0] + "_first.csv"
#     else:
#         raise ValueError("Invalid Parser Type")


def get_save_filepath(parser_type: str, dir_path: str, filename: str, surface: int):
    if parser_type == "track":
        file_name = os.path.basename(filename)
        file_name = os.path.splitext(file_name)[0] + f"_track_{surface}.csv"
        return os.path.join(dir_path, file_name)
    elif parser_type == "surface":
        file_name = os.path.basename(filename)
        file_name = os.path.splitext(file_name)[0] + f"_surface_{surface}.csv"
        return os.path.join(dir_path, file_name)
    elif parser_type == "first":
        file_name = os.path.basename(filename)
        file_name = os.path.splitext(file_name)[0] + f"_first_{surface}.csv"
        return os.path.join(dir_path, file_name)
    else:
        raise ValueError("Invalid Parser Type")
