import h5py
import numpy as np
import pandas as pd
import re
import yaml
import os
from collections.abc import MutableMapping
from typing import Dict, List
from .exceptions import *


##############################################################
class ImarisDataParser:
    def __init__(self, ims_file_path: str) -> None:
        pass

    def load_ims(self, ims_file_path: str) -> h5py.File:
        """
        loads a imaris file with .ims extension using h5py

        Args:
            ims_file_path (str): path to .ims file

        Returns:
            h5py.File: returns the data contained within the .ims file
        """
        return h5py.File(ims_file_path, "r+")

    def get_object_names(full_data_file: h5py.File, search_for: str) -> List:
        """
        extracts the object names that we are interested in.

        Args:
            full_data_file (h5py.File): full imaris file in h5py File format
            search_for (str): string containing full or partial filename to search for

        Returns:
            list: a list of all the object names that match search_for parameter
        """
        try:
            values = full_data_file.get("Scene8").get("Content").keys()
            storage = list()
            for item in values:
                if len(re.findall(search_for, item)):
                    storage.append(item)
            return storage
        except AttributeError:
            raise NoSurfaceException

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
        obj_specific_data = full_data_file.get("Scene8").get("Content").get(object_name)

        # rearrange data
        statistics_name = np.asarray(obj_specific_data["StatisticsType"])
        statistics_name = pd.DataFrame(statistics_name)

        # extract statistics names
        stats_name = statistics_name["Name"]

        # extract statistics ID names
        stats_type = statistics_name["ID"]

        # combine stats type and stats names
        return dict(zip(stats_type, stats_name))

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
        stats_dict = {
            key: str(stats_dict[key]).strip("b")[1:-1] for key in sorted_dict_key
        }

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

    def get_track_ids(full_data_file: h5py.File, object_name: str) -> pd.Series:
        """
        Gets all the track ids for a given surface
        Args:
            full_data_file (h5py.File): _description_
            object_name (str): _description_
        """
        try:
            track_ids = pd.DataFrame(
                np.asarray(
                    full_data_file.get("Scene8").get("Content")[object_name]["Track0"]
                )
            )
            return track_ids["ID"]
        except KeyError:
            raise NoTrackException

    # make @property
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
        return list(inverted_stats_names.keys())
        # print(f"Saved Stats Categories in directory: {os.getcwd()}")
