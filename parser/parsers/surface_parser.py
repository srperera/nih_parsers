import gc
import os
import ray
import pandas as pd
from copy import deepcopy
from functools import partial
from typing import Dict, List

from .parser_base import Parser
from imaris.imaris import ImarisDataObject

# Notes:
"""
* if init surface_id = -1 it will load all surfaces into memory
    * we can use a int surface_id value in extract or inspect to access info
    
* if init surface_id is given a specific value, it will only load that one surface info to memory
    * we have to use int surface_id = 0 in extract or inspect to access the corressponding data
    
* above two steps are done to help improve memory allocation during parallel execution.
"""


#############################################################################
@ray.remote
class SurfaceParserDistributed(Parser):
    """
    Extracts Surface Level Information From Imaris File

    Args:
        Parser (ABCMeta): Parser Abstract Base Class
    """

    def __init__(
        self,
        ims_file_path: str,
        surface_id: int = -1,
        save_dir: str = None,
    ) -> None:
        """
        Args:
            * ims_file_path (str): path to .ims file
            * surface_id (int, optional): specific surface id to extract info from. Defaults to -1.
                If none is provided it will default to -1 where we extract and save to memory info
                from all surfaces. If running in parallel its better to specify the surface
                so we only extract and store limited amount of information.
            * save_dir (str, optional): directory to save csv to. Defaults to None.
        """
        # TODO set up such that we can pass in a path of stats the user wants and we filter final csv accordingly
        self.ims_file_path = ims_file_path
        self.surface_id = surface_id
        self.save_dir = save_dir
        self.ims = ImarisDataObject(self.ims_file_path)
        self._configure_instance(surface_id=surface_id)

        del self.ims
        gc.collect()

    def _configure_instance(self, surface_id: int) -> None:
        """
        * Extracts relevant information from ims object and
        instantiates it as instance variables for fast recall.

        Args:
            surface_id (int): specific surface id to extract info from. ZERO INDEXED

        Currently Extracts:
            - all the surface names -- List
            - all the stats_names -- {id: pd.DataFrame}
            - all the stats values -- {id: pd.DataFrame}
            - all the factor info -- {id: pd.DataFrame}
        """
        # extract all information and saves it as a instance var
        if surface_id == -1:  # configure all available surfaces
            self.surface_names = self.ims.get_object_names("Surface")
        else:  # grab the surface we care about,
            self.surface_names = self.ims.get_object_names("Surface")
            if (surface_id >= 0) and (surface_id < len(self.surface_names)):
                self.surface_names = [self.surface_names[surface_id]]
            elif surface_id >= len(self.surface_names):
                raise ValueError(
                    f"surface_id {surface_id} exceeds number of surfaces available"
                )
            else:  # some currently unknown error
                raise NotImplementedError

        assert type(self.surface_names) == list, "surface_names should be a list"

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

    def _organize_stats(self, stats_values: pd.DataFrame) -> Dict:
        """
        * Organized the data such that it looks like
        {ID_Object: {Stats Name: Value}}

        Args:
            * surface_stats_values (pd.DataFrame): a single dataframe
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
        surface_id: int,
    ) -> None:
        # TODO: instead of surface_id, see if we can insert the REAL surface name
        # To do this we can grab all the surface names, and find the one that is missing
        # as the current surface_id. because one is always missing from the factor list
        # and the one that is missing is the name we want.
        # a function to write csv information to disk
        # get save_dir/original_filename.csv
        ims_filename = os.path.basename(self.ims_file_path).split(".")[0]
        ims_filename = f"{ims_filename}_surface_{(surface_id + 1)}.csv"
        save_filepath = os.path.join(save_dir, ims_filename)
        dataframe.to_csv(save_filepath)

        # store ims_filename
        self.ims_filename = ims_filename

    def _process(self, surface_id: int) -> None:
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
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)
        stat_names = self._update_surface_info(stats_names=stat_names, factor=factor)

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
        # for surface parser need to filter out track id information
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

    def _update_surface_info(
        self,
        stats_names: pd.DataFrame,
        factor: pd.DataFrame,
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
        def _get_surface_name(row_info, factor: pd.DataFrame):
            factor_id = row_info["ID_FactorList"]  # factor id
            name = row_info["Name"]  # stat name

            # filter factor to only include items related to Channel
            surface_info = factor[factor["Name"] == "Surfaces"]

            # main logic to select the right channel given the factor id
            if factor_id in surface_info["ID_List"].to_list():
                surface = surface_info[surface_info["ID_List"] == factor_id][
                    "Level"
                ].item()
                surface = f"{name}_{surface}"
                return surface
            # if factor id is not in the channel list no channel info is needed
            else:
                return name

        # create partial
        get_surface_name_partial = partial(_get_surface_name, factor=factor)

        # update stats name with the newly mapped stats names values
        stats_names["Name"] = stats_names.apply(func=get_surface_name_partial, axis=1)

        return stats_names

    def _drop_unwanted_stats(self):
        """
        Drops stats not contained in the user defined list of stats names
        from the final csv before saving to disk.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def inspect(self, surface_id: int) -> Dict:
        """
        Runs a single end to end parser pipeline on a single surface
        and returns all components as a dict.
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
        # check 1
        if (self.surface_id != -1) and (surface_id != 0):
            raise ValueError(
                f"class is initialized with 1 surface, surface_id should be set to 0"
            )

        # check 2
        if surface_id > len(self.surface_names):
            raise ValueError(
                f"surface_id {surface_id} exceeds number of surfaces available {len(self.surface_names)}"
            )

        # dict to hold all values to be returned for inspection
        storage = {}

        # gather info for current surface
        surface_name = self.surface_names[surface_id]
        storage["surface_name"] = surface_name
        stat_names = self.stats_names.get(surface_id)
        storage["stat_names_raw"] = stat_names
        stat_values = self.stats_values.get(surface_id)
        storage["stat_values_raw"] = stat_values
        object_id = self.object_ids.get(surface_id)
        storage["object_id"] = object_id
        factor = self.factors.get(surface_id)
        storage["factor"] = factor

        # update channel and surface names
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)
        storage["stat_names_channel_info"] = stat_names
        stat_names = self._update_surface_info(stats_names=stat_names, factor=factor)
        storage["stat_names_surface_info"] = stat_names

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

    def get_surface_stats_info(self, surface_id: int) -> List[str]:
        """Returns all the stats information in a given surface id

        Args:
            surface_id (int): _description_

        Returns:
            List[str]: _description_
        """
        # check 1
        if (self.surface_id != -1) and (surface_id != 0):
            raise ValueError(
                f"class is initialized with 1 surface, surface_id should be set to 0 or None"
            )

        # check 2
        if surface_id > len(self.surface_names):
            raise ValueError(
                f"surface_id {surface_id} exceeds number of surfaces available {len(self.surface_names)}"
            )

        # gather info for current surface
        surface_name = self.surface_names[surface_id]
        stat_names = self.stats_names.get(surface_id)
        stat_values = self.stats_values.get(surface_id)
        object_id = self.object_ids.get(surface_id)
        factor = self.factors.get(surface_id)

        # update channel and surface names
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)
        stat_names = self._update_surface_info(stats_names=stat_names, factor=factor)

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

    def extract_and_save(self, surface_id: int, save_dir: str = None) -> None:
        # this function is the funtion that gets called externally
        # we can have this function as a ray method to help with distributed execution
        # check 1
        if (self.surface_id != -1) and (surface_id != 0):
            raise ValueError(
                f"class is initialized with 1 surface, surface_id should be set to 0"
            )

        # check 2
        if surface_id > len(self.surface_names):
            raise ValueError(
                f"surface_id {surface_id} exceeds number of surfaces available {len(self.surface_names)}"
            )

        # process surface
        dataframe = self._process(surface_id)

        # adjust surface_id based on init mode
        # save surface
        save_dir = save_dir if save_dir else self.save_dir
        if self.surface_id == -1:
            self._save_csv(dataframe, save_dir, surface_id=surface_id)
        else:
            self._save_csv(dataframe, save_dir, surface_id=self.surface_id)

        print(f"[info] -- finished: {self.ims_filename}")


#############################################################################
