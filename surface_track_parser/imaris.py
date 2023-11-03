import h5py
import numpy as np
import pandas as pd
import re
import yaml
import os
from typing import Dict, List
from exceptions import *


#########################################################################################
class ImarisDataObject:
    """
    A class to interact with imaris (.ims) files
    """

    def __init__(self, ims_file_path: str) -> None:
        self.data = self.load_ims(ims_file_path)

    def load_ims(self, ims_file_path: str) -> h5py.File:
        """
        loads a imaris file with .ims extension using h5py

        Args:
            ims_file_path (str): path to .ims file

        Returns:
            h5py.File: returns the data contained within the .ims file
        """
        return h5py.File(ims_file_path, "r+")

    def get_object_names(self, object_name: str) -> List:
        """
        extracts the object names that we are interested in.
        objects are usually surfaces( is this always true?)

        Args:
            full_data_file (h5py.File): full imaris file in h5py File format
            object_name (str): string containing full or partial filename to search for

        Returns:
            list: a list of all the object names that match search_for parameter
        """
        try:
            values = self.data.get("Scene8").get("Content").keys()
            storage = list()
            for item in values:
                if len(re.findall(object_name, item)):
                    storage.append(item)
            return storage
        except AttributeError:
            raise NoSurfaceException

    def get_stats_names(self, object_name: str) -> Dict:
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
        obj_specific_data = self.data.get("Scene8").get("Content").get(object_name)

        # rearrange data
        statistics_name = np.asarray(obj_specific_data["StatisticsType"])
        statistics_name = pd.DataFrame(statistics_name)

        # extract statistics names
        stats_name = statistics_name["Name"]

        # extract statistics ID names
        stats_type = statistics_name["ID"]

        # combine stats type and stats names
        return dict(zip(stats_type, stats_name))

    def get_stats_values(self, object_name: str) -> pd.DataFrame:
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
        obj_specific_stats = self.data.get("Scene8").get("Content")[object_name][
            "StatisticsValue"
        ]
        obj_specific_stats = np.asarray(obj_specific_stats)
        return pd.DataFrame(obj_specific_stats)

    def get_track_ids(self, object_name: str) -> pd.Series:
        """
        Gets all the track ids for a given surface

        Args:
            full_data_file (h5py.File): _description_
            object_name (str): _description_

        Returns:
            pd.Series: Track ids for given surface
        """
        try:
            track_ids = pd.DataFrame(
                np.asarray(
                    self.data.get("Scene8").get("Content")[object_name]["Track0"]
                )
            )
            return track_ids["ID"]
        except KeyError:
            raise NoTrackException

    def get_real_surface_names(self, surface_name: str) -> List:
        """Gets all the real surface names for given surface name

        Args:
            surface_name (str): example: MegaSurface0

        Returns:
            List: _description_
        """
        arr = self.data.get("Scene8").get("Content").get(surface_name).get("Factor")
        surface_names = []
        for item in arr:
            if len(re.findall(b"Surface", item)):
                surface_names.append(item[-1])
        return surface_names
