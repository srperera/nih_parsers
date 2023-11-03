import h5py
import numpy as np
import pandas as pd
import re
import yaml
import os
from collections.abc import MutableMapping
from typing import Dict, List
from exceptions import *
from copy import deepcopy
import collections
import polars


#####################################################################
class SurfaceParser:
    def __init__(self, ims_file_path: str, config) -> None:
        self.data_obj = ImarisDataObject(ims_file_path)
        self.surface_names = self.data_obj.get_object_names(object_name="Surface")
        self.config = config

    def group_stats(self, surface_stats_values: pd.DataFrame) -> Dict:
        """Groups the data such that it looks like
        {ID_Object: {ID_StatisticsType: Value}}

        Args:
            surface_stats_values (pd.DataFrame): a single dataframe
            that contains the statistics for a single surface

        Returns:
            Dict: _description_
        """
        grouped_stats = (
            surface_stats_values.groupby("ID_Object")[["ID_StatisticsType", "Value"]]
            .apply(lambda x: x.set_index("ID_StatisticsType").to_dict(orient="dict"))
            .to_dict()
        )
        grouped_stats = {k: v["Value"] for k, v in grouped_stats.items()}
        return grouped_stats

    def get_filtered_stats_list(self, column_names: Dict, surface_id: int) -> Dict:
        """
        Helper function to generate lists to remove

        Args:
            column_names (Dict): _description_
            surface_id (int): _description_

        Returns:
            Dict: _description_
        """
        filtered_column_names = list()
        for item in column_names.values():
            if len(re.findall(b"Track", item)):
                filtered_column_names.append(item)

        # drop columns related to trackids
        surface_name = self.surface_names[surface_id]
        track_ids = self.data_obj.get_track_ids(surface_name).to_list()

        # filtered stats list
        filtered_stats_list = {
            "track_column_names": filtered_column_names,
            "track_ids": track_ids,
        }

        return filtered_stats_list

    def generate_csv(
        self, data_dict: Dict, column_names: Dict, filter_info: Dict
    ) -> pd.DataFrame:
        # converts the dictionary to a dataframe

        # convert to dataframe
        dataframe = pd.DataFrame.from_dict(data_dict).transpose()

        # update column info
        dataframe = dataframe.rename(column_names, axis=1)
        dataframe["ID"] = dataframe.index

        # filter out and stats columns we dont need
        dataframe = dataframe.drop(labels=filter_info["track_column_names"], axis=1)
        dataframe = dataframe.drop(labels=filter_info["track_ids"], axis=0)

        return dataframe

    def save_csv(self, data: pd.DataFrame, surface_id: int) -> None:
        """
        Saves given dataframe to the given data path

        Args:
            data (pd.DataFrame): Data to be saved
            save_path (str): Directory to save to
        """
        raise NotImplementedError

    def process(self, surface_id: int) -> Dict:
        # get surface names
        surface_name = self.surface_names[surface_id]  # should be 0 indexed

        # organize stats data
        available_stats_names = self.data_obj.get_stats_names(surface_name)
        available_stats_values = self.data_obj.get_stats_values(surface_name)
        grouped_stats = self.group_stats(available_stats_values)
        # since we are working on surface should we drop the track level
        # information?

        # generate column names
        stats_ids = available_stats_values["ID_StatisticsType"].unique()
        column_names = {key: available_stats_names[key] for key in stats_ids}

        # processed out
        processed_out = {"grouped_stats": grouped_stats, "column_names": column_names}

        return processed_out

    def extract_and_save(self, surface_id: int) -> None:
        # main access point
        # processes the whole dataset
        processed_out = self.process(surface_id=surface_id)
        filter_info = self.get_filtered_stats_list(
            column_names=processed_out["column_names"],
            surface_id=surface_id,
        )
        generated_csv = self.generate_csv(
            data_dict=processed_out["grouped_stats"],
            column_names=processed_out["column_names"],
            filter_info=filter_info,
        )
        # self.save_csv(data=generated_csv, surface_id=surface_id)
        return generated_csv

    def get_available_stats(self):
        raise NotImplementedError

    def generate_summary(self):
        raise NotImplementedError
