import gc
import os
import ray
import pandas as pd
from functools import partial
from typing import Dict, List

from .parser_base import Parser
from imaris.imaris import ImarisDataObject
import time

# Notes:
"""
* if init filament_id = -1 it will load all filaments into memory
    * we can use a int filament_id value in extract or inspect to access info
    
* if init filament_id is given a specific value, it will only load that one filament info to memory
    * we have to use int filament_id = 0 in extract or inspect to access the corressponding data
    
* above two steps are done to help improve memory allocation during parallel execution.
"""


#############################################################################
@ray.remote
class FilamentParserDistributed(Parser):
    """
    Extracts Filament Level Information From Imaris File

    Args:
        Parser (ABCMeta): Parser Abstract Base Class
    """

    def __init__(
        self,
        ims_file_path: str,
        filament_id: int = -1,
        save_dir: str = None,
    ) -> None:
        """
        Args:
            * ims_file_path (str): path to .ims file
            * filament_id (int, optional): specific filament id to extract info from. Defaults to -1.
                If none is provided it will default to -1 where we extract and save to memory info
                from all filaments. If running in parallel its better to specify the filament
                so we only extract and store limited amount of information.
            * save_dir (str, optional): directory to save csv to. Defaults to None.
        """
        # TODO set up such that we can pass in a path of stats the user wants and we filter final csv accordingly
        self.ims_file_path = ims_file_path
        self.filament_id = filament_id
        self.save_dir = save_dir
        self.ims = ImarisDataObject(self.ims_file_path)
        self._configure_instance(filament_id=filament_id)

        del self.ims
        gc.collect()

    def _configure_instance(self, filament_id: int) -> None:
        """
        * Extracts relevant information from ims object and
        instantiates it as instance variables for fast recall.

        Args:
            filament_id (int): specific filament id to extract info from. ZERO INDEXED

        Currently Extracts:
            - all the filament names -- List
            - all the stats_names -- {id: pd.DataFrame}
            - all the stats values -- {id: pd.DataFrame}
            - all the factor info -- {id: pd.DataFrame}
        """
        # extract all information and saves it as a instance var
        if filament_id == -1:  # configure all available filaments
            self.filament_names = self.ims.get_object_names("Filament")
        else:  # grab the filament we care about,
            self.filament_names = self.ims.get_object_names("Filament")
            if (filament_id >= 0) and (filament_id < len(self.filament_names)):
                self.filament_names = [self.filament_names[filament_id]]
            elif filament_id >= len(self.filament_names):
                raise ValueError(
                    f"filament_id {filament_id} exceeds number of filaments available"
                )
            else:  # some currently unknown error
                raise NotImplementedError

        assert type(self.filament_names) == list, "filament_names should be a list"

        # get all the stats names for every filament {surf_id: stats_name_df}
        self.stats_names = {
            filament_id: self.ims.get_stats_names(filament_name)
            for filament_id, filament_name in enumerate(self.filament_names)
        }

        # get all the stats values for every filament {surf_id: stats_values_df}
        self.stats_values = {
            filament_id: self.ims.get_stats_values(filament_name)
            for filament_id, filament_name in enumerate(self.filament_names)
        }

        # get all the factor table info for every filament {surf_id: factor_df}
        self.factors = {
            filament_id: self.ims.get_object_factor(filament_name)
            for filament_id, filament_name in enumerate(self.filament_names)
        }

        # get all the factor table info for every filament {surf_id: factor_df}
        self.object_ids = {
            filament_id: self.ims.get_object_ids(filament_name, use_stats_data=True)
            for filament_id, filament_name in enumerate(self.filament_names)
        }

    def _organize_stats(self, stats_values: pd.DataFrame) -> Dict:
        """
        * Organized the data such that it looks like
        {ID_Object: {Stats Name: Value}}

        Args:
            * stats_values (pd.DataFrame): a single dataframe
            that contains the statistics for a single filament

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

    def _generate_csv(
        self,
        stats_values: Dict,
        stat_names: pd.DataFrame,
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

    def _save_csv(
        self,
        dataframe: pd.DataFrame,
        save_dir: str,
        filament_id: int,
    ) -> None:
        # TODO: instead of filament_id, see if we can insert the REAL filament name
        # To do this we can grab all the filament names, and find the one that is missing
        # as the current filament_id. because one is always missing from the factor list
        # and the one that is missing is the name we want.
        # a function to write csv information to disk
        # get save_dir/original_filename.csv
        ims_filename = os.path.basename(self.ims_file_path).split(".")[0]
        ims_filename = f"{ims_filename}_filament_{(filament_id + 1)}.csv"
        save_filepath = os.path.join(save_dir, ims_filename)
        dataframe.to_csv(save_filepath)

        # store ims_filename
        self.ims_filename = ims_filename

    def _process(self, filament_id: int) -> None:
        """
        Runs a single end to end parser pipeline on a single filament
        Steps:
            - get stat names for a single filament
            - get stat values for a single filament
            - filter stat values to keep only track ids
            - filter stats values to remove track level stat information
            - rename certian columns (if needed)(need a custom func for this to add channel info)
            - organize the filtered stats
            - generate csv
            - save csv

        Args:
            filament_id (int): _description_
        """
        # gather info for current filament
        filament_name = self.filament_names[filament_id]
        stat_names = self.stats_names.get(filament_id)
        stat_values = self.stats_values.get(filament_id)
        object_id = self.object_ids.get(filament_id)
        factor = self.factors.get(filament_id)

        # update channel information
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)
        # update image level information
        stat_names = self._update_image_level_info(
            stats_names=stat_names, factor=factor
        )
        # update image depth level information
        stat_names = self._update_depth_level_info(
            stats_names=stat_names, factor=factor
        )

        # update level information
        stat_names = self._update_level_info(stats_names=stat_names, factor=factor)

        # filter stats values by object ids (ie: ignore info related to trackids)
        stat_values = self._filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )

        # organize stats value (most compute used here)
        organized_stats = self._organize_stats(stat_values)

        # generate csv
        stats_df = self._generate_csv(organized_stats, stat_names=stat_names)

        return stats_df

    def _filter_stats(
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
        # for filament parser need to filter out track id information
        # and statistics related to track information.
        for col_names, values in zip(filter_col_names, filter_values):
            stats_values = stats_values[stats_values[col_names].isin(values=values)]

        return stats_values

    def _update_channel_info(
        self,
        stats_names: pd.DataFrame,
        factor: pd.DataFrame,
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
        def _get_channel_id(row_info, factor: pd.DataFrame):
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
        get_channel_id_partial = partial(_get_channel_id, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_channel_id_partial, axis=1)

        return stats_names

    def _update_level_info(
        self,
        stats_names: pd.DataFrame,
        factor: pd.DataFrame,
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
        def _get_channel_id(row_info, factor: pd.DataFrame):
            factor_id = row_info["ID_FactorList"]  # factor id
            name = row_info["Name"]  # stat name

            # filter factor to only include items related to Channel
            channel_info = factor[factor["Name"] == "Level"]

            # main logic to select the right channel given the factor id
            if factor_id in channel_info["ID_List"].to_list():
                channel = channel_info[channel_info["ID_List"] == factor_id][
                    "Level"
                ].item()
                return f"{name} Level_{channel}"
            # if factor id is not in the channel list no channel info is needed
            else:
                return name

        # create partial
        get_channel_id_partial = partial(_get_channel_id, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_channel_id_partial, axis=1)

        return stats_names

    def _update_image_level_info(
        self,
        stats_names: pd.DataFrame,
        factor: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Updates the filament name information for the relavent rows
        based on th ID_FactorList information in stats_names

        Args:
            stats_names (pd.DataFrame): _description_
            factor (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # create function get channel number from a pandas row from stats_names
        # inner func
        def _get_channel_id(row_info, factor: pd.DataFrame):
            factor_id = row_info["ID_FactorList"]  # factor id
            name = row_info["Name"]  # stat name

            # filter factor to only include items related to Channel
            channel_info = factor[factor["Name"] == "Image"]

            # main logic to select the right channel given the factor id
            if factor_id in channel_info["ID_List"].to_list():
                channel = channel_info[channel_info["ID_List"] == factor_id][
                    "Level"
                ].item()
                return f"{name} {channel}"  # channel = "Image 1"
            # if factor id is not in the channel list no channel info is needed
            else:
                return name

        # create partial
        get_channel_id_partial = partial(_get_channel_id, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_channel_id_partial, axis=1)

        return stats_names

    def _update_depth_level_info(
        self,
        stats_names: pd.DataFrame,
        factor: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Updates the filament name information for the relavent rows
        based on th ID_FactorList information in stats_names

        Args:
            stats_names (pd.DataFrame): _description_
            factor (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # create function get channel number from a pandas row from stats_names
        # inner func
        def _get_channel_id(row_info, factor: pd.DataFrame):
            factor_id = row_info["ID_FactorList"]  # factor id
            name = row_info["Name"]  # stat name

            # filter factor to only include items related to Channel
            channel_info = factor[factor["Name"] == "Depth"]

            # main logic to select the right channel given the factor id
            if factor_id in channel_info["ID_List"].to_list():
                channel = channel_info[channel_info["ID_List"] == factor_id][
                    "Level"
                ].item()
                return f"{name} Depth_{channel}"  # channel = "Image 1"
            # if factor id is not in the channel list no channel info is needed
            else:
                return name

        # create partial
        get_channel_id_partial = partial(_get_channel_id, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_channel_id_partial, axis=1)

        return stats_names

    def _drop_unwanted_stats(self):
        """
        Drops stats not contained in the user defined list of stats names
        from the final csv before saving to disk.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def inspect(self, filament_id: int) -> Dict:
        """
        Runs a single end to end parser pipeline on a single filament
        and returns all components as a dict.
        Steps:
            - get stat names for a single filament
            - get stat values for a single filament
            - filter stat values to keep only track ids
            - filter stats values to remove track level stat information
            - rename certian columns (if needed)(need a custom func for this to add channel info)
            - organize the filtered stats
            - generate csv
            - save csv

        Args:
            filament_id (int): _description_
        """
        # self._configure_instance(filament_id=filament_id)
        # del self.ims
        # gc.collect()

        # check 1
        if (self.filament_id != -1) and (filament_id != 0):
            raise ValueError(
                f"class is initialized with 1 filament, filament_id should be set to 0"
            )

        # check 2
        if filament_id > len(self.filament_names):
            raise ValueError(
                f"filament_id {filament_id} exceeds number of filaments available {len(self.filament_names)}"
            )

        # dict to hold all values to be returned for inspection
        storage = {}

        # gather info for current filament
        filament_name = self.filament_names[filament_id]
        storage["filament_name"] = filament_name
        stat_names = self.stats_names.get(filament_id)
        storage["stat_names_raw"] = stat_names
        stat_values = self.stats_values.get(filament_id)
        storage["stat_values_raw"] = stat_values
        object_id = self.object_ids.get(filament_id)
        storage["object_id"] = object_id
        factor = self.factors.get(filament_id)
        storage["factor"] = factor

        # update channel information
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)
        storage["stat_names_channel_info"] = stat_names

        # update image level information
        stat_names = self._update_image_level_info(
            stats_names=stat_names, factor=factor
        )
        storage["stat_names_image_info"] = stat_names

        # update image depth level information
        stat_names = self._update_depth_level_info(
            stats_names=stat_names, factor=factor
        )
        storage["stat_names_depth_info"] = stat_names

        # update level information
        stat_names = self._update_level_info(stats_names=stat_names, factor=factor)
        storage["stat_names_filament_info"] = stat_names

        # filter stats values by object ids (ie: ignore info related to trackids)
        stat_values = self._filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )
        storage["stat_values_filtered"] = stat_values

        # organize stats value (most compute used here)
        organized_stats = self._organize_stats(stat_values)
        storage["organized_stats"] = organized_stats

        # generate csv
        stats_df = self._generate_csv(organized_stats, stat_names=stat_names)
        storage["stats_df"] = stats_df

        return storage

    def get_filament_stats_info(self, filament_id: int) -> List[str]:
        """Returns all the stats information in a given filament id

        Args:
            filament_id (int): _description_

        Returns:
            List[str]: _description_
        """
        # check 1
        if (self.filament_id != -1) and (filament_id != 0):
            raise ValueError(
                f"class is initialized with 1 filament, filament_id should be set to 0 or None"
            )

        # check 2
        if filament_id > len(self.filament_names):
            raise ValueError(
                f"filament_id {filament_id} exceeds number of filaments available {len(self.filament_names)}"
            )

        # gather info for current filament
        filament_name = self.filament_names[filament_id]
        stat_names = self.stats_names.get(filament_id)
        stat_values = self.stats_values.get(filament_id)
        object_id = self.object_ids.get(filament_id)
        factor = self.factors.get(filament_id)

        # update channel information
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)

        # update image level information
        stat_names = self._update_image_level_info(
            stats_names=stat_names, factor=factor
        )

        # update image depth level information
        stat_names = self._update_depth_level_info(
            stats_names=stat_names, factor=factor
        )

        # update level information
        stat_names = self._update_level_info(stats_names=stat_names, factor=factor)

        # filter stats values by object ids (ie: ignore info related to trackids)
        stat_values = self._filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )

        # create dict that maps stat id to name
        # key = id, value = name
        stats_dict = dict(zip(stat_names["ID"], stat_names["Name"]))

        # keep only unique for display
        # we can also modify this line to return a dict so we can see id and name
        available_stats_names = [
            stats_dict[ids] for ids in stat_values["ID_StatisticsType"].unique()
        ]

        return stats_dict, available_stats_names

    def extract_and_save(self, filament_id: int, save_dir: str = None) -> None:
        # this function is the funtion that gets called externally
        # we can have this function as a ray method to help with distributed execution
        # self._configure_instance(filament_id=filament_id)
        # del self.ims
        # gc.collect()

        # check 1
        if (self.filament_id != -1) and (filament_id != 0):
            raise ValueError(
                f"class is initialized with 1 filament, filament_id should be set to 0"
            )

        # check 2
        if filament_id > len(self.filament_names):
            raise ValueError(
                f"filament_id {filament_id} exceeds number of filaments available {len(self.filament_names)}"
            )

        # process filament
        dataframe = self._process(filament_id)

        # adjust filament_id based on init mode
        # save filament
        save_dir = save_dir if save_dir else self.save_dir
        if self.filament_id == -1:
            self._save_csv(dataframe, save_dir, filament_id=filament_id)
        else:
            self._save_csv(dataframe, save_dir, filament_id=self.filament_id)

        print(f"[info] -- finished: {self.ims_filename}")


#############################################################################
