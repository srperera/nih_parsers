import h5py
import numpy as np
import pandas as pd
import re
import yaml
import os
from collections.abc import MutableMapping
from typing import Dict, List, Tuple
from exceptions import *
from copy import deepcopy
import collections
from parser_base import Parser
from imaris import ImarisDataObject
import ray
from functools import partial
import time
import polars as pl


###################################################################################
# @ray.remote
class TimeStepSurfaceParser(Parser):
    """
    Extracts Surface information for a given time step Imaris File

    Args:
        Parser (ABCMeta): Parser Abstract Base Class
    """

    def __init__(self, ims_file_path: str, time_step: float = 1.0) -> None:
        self.time_step = time_step
        self.ims_file_path = ims_file_path
        self.ims = ImarisDataObject(self.ims_file_path)
        self.configure_instance()

    def configure_instance(self) -> None:
        """
        Extracts relevant information from ims object and
        instantiates it as instance variables for fast recall.

        Currently Extracts:
            - all the surface names -- List
            - all the stats_names -- {id: pd.DataFrame}
            - all the stats values -- {id: pd.DataFrame}
            - all the factor info -- {id: pd.DataFrame}
        """
        # TODO: check to ensure surfaces exist or raise error
        # extract all information and saves it as a instance var
        self.surface_names = self.ims.get_object_names("Surface")

        # get all the stats names for every surface {surf_id: stats_name_df}
        self.stats_names = {
            surface_id: self.ims.get_stats_names(surface_name)
            for surface_id, surface_name in enumerate(self.surface_names)
        }

        # get all the stats values for every surface {surf_id: stats_values_df}
        self.stats_values = {
            surface_id: self.ims.get_stats_values(surface_name)
            for surface_id, surface_name in enumerate(self.surface_names)
        }

        # get all the factor table info for every surface {surf_id: factor_df}
        self.factors = {
            surface_id: self.ims.get_object_factor(surface_name)
            for surface_id, surface_name in enumerate(self.surface_names)
        }

        # get all the factor table info for every surface {surf_id: factor_df}
        self.object_ids = {
            surface_id: self.ims.get_object_ids(surface_name)
            for surface_id, surface_name in enumerate(self.surface_names)
        }

    def organize_stats(self, stats_values: pd.DataFrame) -> Dict:
        """Organized the data such that it looks like
        {ID_Object: {Stats Name: Value}}

        Args:
            surface_stats_values (pd.DataFrame): a single dataframe
            that contains the statistics for a single surface

        Returns:
            Dict: _description_
        """
        grouped_stats = (
            stats_values.groupby("ID_Object")[["ID_StatisticsType", "Value"]]
            .apply(lambda x: x.set_index("ID_StatisticsType").to_dict(orient="dict"))
            .to_dict()
        )
        grouped_stats = {k: v["Value"] for k, v in grouped_stats.items()}
        return grouped_stats

    def generate_csv(
        self, stats_values: Dict, stat_names: pd.DataFrame
    ) -> pd.DataFrame:
        """_summary_

        Args:
            organized_stats (Dict): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # create a dict that maps stat_id to stat_name
        column_names_dict = dict(zip(stat_names["ID"], stat_names["Name"]))
        dataframe = pd.DataFrame(stats_values).transpose()

        # replaces id columns with respective stat name and add idx
        dataframe = dataframe.rename(column_names_dict, axis=1)
        dataframe["Object_ID"] = dataframe.index
        return dataframe

    def save_csv(self):
        # a function to write csv information to disk
        pass

    def process(self, surface_id: int) -> None:
        """
        Runs a single end to end parser pipeline on a single surface
        Steps:
            - get stat names for a single surface
            - get stat values for a single surface
            - filter stat values to keep only track ids
            - filter stats values to remove track level stat information
            - rename certian columns (if needed)(need a custom func for this to add channel info)
            - organize the filtered stats
            - generate csv
            - save csv

        Args:
            surface_id (int): _description_
        """
        # gather info for current surface
        start = time.perf_counter()
        surface_name = self.surface_names[surface_id]
        print(f"surface_name: {time.perf_counter() - start}")
        stat_names = self.stats_names.get(surface_id)
        print(f"stat_names: {time.perf_counter() - start}")
        stat_values = self.stats_values.get(surface_id)
        print(f"stat_values: {time.perf_counter() - start}")
        track_id = self.object_ids.get(surface_id)
        print(f"object_id: {time.perf_counter() - start}")
        factor = self.factors.get(surface_id)
        print(f"factor: {time.perf_counter() - start}")

        # update channel and surface names
        stat_names = self.update_channel_info(stats_names=stat_names, factor=factor)
        print(f"stat_names_channel: {time.perf_counter() - start}")
        stat_names = self.update_surface_info(stats_names=stat_names, factor=factor)
        print(f"stat_names_surfaces: {time.perf_counter() - start}")

        # filter stats values by object ids and time index = self.time_step
        time_index_id = stat_names[stat_names["Name"] == "Time Index"]["ID"]
        time_index_id = time_index_id.iloc[0].item()
        filtered_stat_values = self.filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object", "ID_StatisticsType", "Value"],
            filter_values=[
                track_id,
                pd.Series([time_index_id]),
                pd.Series([self.time_step]),
            ],
        )
        print(f"filtered_stat_values: {time.perf_counter() - start}")

        # organize stats values
        organized_stats = self.organize_stats(filtered_stat_values)
        print(f"organized_stats: {time.perf_counter() - start}")

        # generate csv
        stats_df = self.generate_csv(organized_stats, stat_names=stat_names)
        print(f"stats_df: {time.perf_counter() - start}")

        return stats_df

    def filter_stats(
        self,
        stats_values: pd.DataFrame,
        filter_col_names: List[str],
        filter_values: List[pd.Series],
    ) -> pd.DataFrame:
        """
        Filters the stats values dataframe. It keeps information
        from col_names and filter_values that is passed in as arguments.

        Args:
            stats_values (pd.DataFrame): _description_
            filter_col_name (str): name of the column we want to use to filter
            filter_values (str): values that we want to keep

        Returns:
            pd.DataFrame: _description_
        """
        # for first surface parser we apply 3 filters
        # object, statistisc type, and value filters
        # to get the object ids that belong to the first time setp
        stats_values_copy = deepcopy(stats_values)
        for col_names, values in zip(filter_col_names, filter_values):
            stats_values_copy = stats_values_copy[
                stats_values_copy[col_names].isin(values=values)
            ]

        # filter the original stats values to only contain the
        # information from the first time step
        # TODO: maybe doing this with a deepcopy is a good idea
        # That way we can chain multiple filters get the final object ids
        # and return all the stats for those object ids
        stats_values = stats_values[
            stats_values["ID_Object"].isin(values=stats_values_copy["ID_Object"])
        ]

        return stats_values

    def extract_and_save(self):
        # this function is the funtion that gets called externally
        # we can have this function as a ray method to help with distributed execution
        pass

    def get_available_stat_names(self):
        # interacts with data object and returns requested data for inspection
        pass

    def update_channel_info(
        self, stats_names: pd.DataFrame, factor: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates the channel information for the relavent rows
        based on th ID_FactorList information in stats_names

        Args:
            stats_names (pd.DataFrame): _description_
            factor (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # create function get channel number from a pandas row from stats_names
        # inner func
        def get_channel_id(row_info, factor: pd.DataFrame):
            factor_id = row_info["ID_FactorList"]  # factor id
            name = row_info["Name"]  # stat name

            # filter factor to only include items related to Channel
            channel_info = factor[factor["Name"] == "Channel"]

            # main logic to select the right channel given the factor id
            if factor_id in channel_info["ID_List"].to_list():
                channel = channel_info[channel_info["ID_List"] == factor_id][
                    "Level"
                ].item()
                return f"{name} Channel_{channel}"
            # if factor id is not in the channel list no channel info is needed
            else:
                return name

        # create partial
        get_channel_id_partial = partial(get_channel_id, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_channel_id_partial, axis=1)

        return stats_names

    def update_surface_info(
        self, stats_names: pd.DataFrame, factor: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates the surface name information for the relavent rows
        based on th ID_FactorList information in stats_names

        Args:
            stats_names (pd.DataFrame): _description_
            factor (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # create function get channel number from a pandas row from stats_names
        # inner func
        def get_surface_name(row_info, factor: pd.DataFrame):
            factor_id = row_info["ID_FactorList"]  # factor id
            name = row_info["Name"]  # stat name

            # filter factor to only include items related to Channel
            channel_info = factor[factor["Name"] == "Surfaces"]

            # main logic to select the right channel given the factor id
            if factor_id in channel_info["ID_List"].to_list():
                channel = channel_info[channel_info["ID_List"] == factor_id][
                    "Level"
                ].item()
                return channel
            # if factor id is not in the channel list no channel info is needed
            else:
                return name

        # create partial
        get_surface_name_partial = partial(get_surface_name, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_surface_name_partial, axis=1)

        return stats_names
