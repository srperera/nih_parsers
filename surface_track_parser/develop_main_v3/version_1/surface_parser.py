import pandas as pd
import os
from typing import Dict, List
from exceptions import *
from copy import deepcopy
from parser_base import Parser
from imaris import ImarisDataObject
import ray
from functools import partial


#############################################################################
@ray.remote
class SurfaceParser(Parser):
    """
    Extracts Surface Level Information From Imaris File

    Args:
        Parser (ABCMeta): Parser Abstract Base Class
    """

    def __init__(self, ims_file_path: str) -> None:
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

    def save_csv(self, surface_id: int, dataframe: pd.DataFrame, save_dir: str) -> None:
        # a function to write csv information to disk
        # get save_dir/original_filename.csv
        ims_filename = os.path.basename(self.ims_file_path).split(".")[0]
        ims_filename = f"{ims_filename}_surface_{surface_id}.csv"
        save_filepath = os.path.join(save_dir, ims_filename)
        dataframe.to_csv(save_filepath)

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
        surface_name = self.surface_names[surface_id]
        stat_names = self.stats_names.get(surface_id)
        stat_values = self.stats_values.get(surface_id)
        object_id = self.object_ids.get(surface_id)
        factor = self.factors.get(surface_id)

        # update channel and surface names
        stat_names = self.update_channel_info(stats_names=stat_names, factor=factor)
        stat_names = self.update_surface_info(stats_names=stat_names, factor=factor)

        # filter stats values by object ids (ie: ignore info related to trackids)
        filtered_stat_values = self.filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )

        # organize stats value
        organized_stats = self.organize_stats(filtered_stat_values)

        # generate csv
        stats_df = self.generate_csv(organized_stats, stat_names=stat_names)

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
        # for surface parser need to filter out track id information
        # and statistics related to track information.
        for col_names, values in zip(filter_col_names, filter_values):
            stats_values = stats_values[stats_values[col_names].isin(values=values)]

        return stats_values

    def extract_and_save(self, surface_id: int, save_dir: str) -> None:
        # this function is the funtion that gets called externally
        # we can have this function as a ray method to help with distributed execution
        dataframe = self.process(surface_id)
        self.save_csv(surface_id, dataframe, save_dir)

    def update_stats_with_real_names(
        self, surface_name: str, stats_names: Dict, user_defined_list: List
    ) -> Dict:
        """
        Update the stats names according to the real surface names found
        inside Contents->SurfaceName->Factor

        Args:
            surface_name (str): the name of the surface to extract data from
            stats_names (Dict): stats_names dictionary
            user_defined_list (List): list of stats name given by ...
                ...the user to be replaced by the real surface names

        Returns:
            Dict: stats name dict with the updated surface names
        """
        real_stats_names = self.ims.get_real_surface_names(surface_name)
        filtered_dicts = [
            self.get_filtered_stat_names(stats_names, keyword, exact=True)
            for keyword in user_defined_list
        ]
        for dict in filtered_dicts:
            for idx, (k, _) in enumerate(dict.items()):
                stats_names[k] = real_stats_names[idx]

        return stats_names

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


#############################################################################
