"""
Notes:
    * Tested with Surfaces and Tracks
"""

import re
import os
import h5py
import numpy as np
import pandas as pd
from .exceptions import *
from typing import Dict, List, Union


#########################################################################################
class ImarisDataObject:
    def __init__(self, ims_file_path: str) -> None:
        self.data = self.load_ims(ims_file_path)
        self.filename = os.path.basename(ims_file_path)

    def load_ims(self, ims_file_path: str) -> h5py.File:
        """
        loads a imaris file with .ims extension using h5py

        Args:
            ims_file_path (str): path to .ims file

        Returns:
            h5py.File: returns the data contained within the .ims file
        """
        if os.path.isfile(ims_file_path):
            return h5py.File(ims_file_path, "r")  # r = read_only
        else:
            raise ValueError("Invalid File Path")

    def get_object_names(self, object_name: str) -> List:
        """
        extracts the object names that we are interested in.
        objects are usually surfaces but can be other items such as
        Points, Filaments etc

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
            if object_name == "Surface":
                raise NoSurfaceException

    def get_stats_names(self, object_name: str) -> pd.DataFrame:
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

        # remove byte txt
        statistics_name["Name"] = statistics_name.apply(
            func=lambda x: x["Name"].decode("utf-8"), axis=1
        )

        return statistics_name

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

    def get_track_ids(self, object_name: str) -> Union[pd.Series, None]:
        """
        Gets all the track ids for a given surface

        Args:
            full_data_file (h5py.File): _description_
            object_name (str): _description_

        Returns:
            pd.Series: Track ids for given surface
        """
        track_ids = pd.DataFrame(
            np.asarray(
                self.data.get("Scene8").get("Content").get(object_name).get("Track0")
            )
        )
        if track_ids is None:
            # maybe log a warning?
            return None
        else:
            return track_ids["ID"]

    def get_track_info(self, object_name: str) -> Union[pd.DataFrame, None]:
        """
        Gets all the track infomation for a given surface

        Args:
            full_data_file (h5py.File): _description_
            object_name (str): _description_

        Returns:
            pd.Series: Track ids for given surface
        """
        track_info = pd.DataFrame(
            np.asarray(
                self.data.get("Scene8").get("Content").get(object_name).get("Track0")
            )
        )
        if track_info is None:
            # maybe log a warning?
            return None
        else:
            return track_info

    def get_object_ids(self, object_name: str, **kwargs) -> Union[pd.Series, None]:
        """
        Gets all the object ids for a given surface.

        Args:
            object_name (str): _description_
            use_stats_data: can be passed in as a kwarg in situations
            where the ims file does not contain a specific location
            to extract object_id information. If use_stats_data is True
            object_id information will be extracted from the StatisticsValue
            dataframe.

        Returns:
            pd.Series: Object ids for given surface
        """
        if kwargs.get("use_stats_data"):
            stats = pd.DataFrame(
                np.asarray(
                    self.data.get("Scene8")
                    .get("Content")
                    .get(object_name)
                    .get("StatisticsValue")
                )
            )
            if stats is None:
                return None
            else:
                object_ids = stats[stats["ID_Time"] != -1]["ID_Object"].unique()
                object_ids = np.where(object_ids >= 0, object_ids, 0)
                object_ids = object_ids[np.nonzero(object_ids)[0]]
                object_ids = pd.Series(object_ids)
                if len(object_ids) > 0:
                    return object_ids
                else:
                    return None
        else:
            object_ids = pd.DataFrame(
                np.asarray(
                    self.data.get("Scene8")
                    .get("Content")
                    .get(object_name)
                    .get("SurfaceModel")
                )
            )
            if object_ids is None:
                return None
            else:
                return object_ids["ID"]

    def get_object_factor(self, surface_name: str) -> pd.DataFrame:
        """Gets all the real surface names for given surface name

        Args:
            surface_name (str): example: MegaSurface0

        Returns:
            pd.DataFrame: _description_
        """
        factor = self.data.get("Scene8").get("Content").get(surface_name).get("Factor")
        factor = np.asarray(factor)
        factor = pd.DataFrame(factor)

        # remove byte text
        factor["Name"] = factor.apply(func=lambda x: x["Name"].decode("utf-8"), axis=1)
        factor["Level"] = factor.apply(
            func=lambda x: x["Level"].decode("utf-8"), axis=1
        )

        return factor

    def contains_surfaces(self, object_name: str) -> bool:
        """
        Given a object name ie: MegaSurfaces0 returns if
        surface object data is available.

        Args:
            object_name (str): _description_

        Returns:
            bool: _description_
        """
        surface_data = (
            self.data.get("Scene8").get("Content").get(object_name).get("SurfaceModel")
        )

        # checks to ensure surface data has objects within it
        if (surface_data is not None) and (surface_data.shape[0] > 0):
            return True
        else:
            return False

    def contains_tracks(self, object_name: str) -> bool:
        """
        Given a object name ie: MegaSurfaces0 returns if
        track data is available.

        Args:
            object_name (str): _description_

        Returns:
            bool: _description_
        """
        track_data = (
            self.data.get("Scene8").get("Content").get(object_name).get("Track0")
        )

        # checks to ensure surface data has objects within it
        if (track_data is not None) and (track_data.shape[0] > 0):
            return True
        else:
            return False

    def contains_filaments(self, object_name: str) -> bool:
        """
        Given a object name ie: Filaments0 returns True
        if filaments data exists else False

        Args:
            object_name (str): _description_

        Returns:
            bool: _description_
        """
        filament_data = (
            self.data.get("Scene8")
            .get("Content")
            .get(object_name)
            .get("StatisticsValue")
        )

        # checks to ensure surface data has objects within it
        if (filament_data is not None) and (filament_data.shape[0] > 0):
            return True
        else:
            return False

    def get_track_object_info(self, object_name: str) -> Union[pd.Series, None]:
        """
        Gets all the object ids associated with each track.
        This data can be used to find what objects belong to
        which track.

        Args:
            object_name (str): _description_

        Returns:
            pd.Series: Object ids for given surface
        """
        object_data = pd.DataFrame(
            np.asarray(
                self.data.get("Scene8")
                .get("Content")
                .get(object_name)
                .get("TrackObject0")
            )
        )
        if object_data is not None:
            return object_data
        else:
            return None
