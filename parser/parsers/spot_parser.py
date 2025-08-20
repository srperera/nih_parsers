import gc
import os
import ray
import pandas as pd
from functools import partial
from typing import Dict, List

from .parser_base import Parser
from imaris.imaris import ImarisDataObject

# Notes:


#############################################################################
# @ray.remote
class SpotParserDistributed(Parser):
    """
    Extracts Spot Level Information From Imaris File

    Args:
        Parser (ABCMeta): Parser Abstract Base Class
    """

    def __init__(
        self,
        ims_file_path: str,
        spot_id: int = -1,
        save_dir: str = None,
    ) -> None:
        # TODO set up such that we can pass in a path of stats the user wants and we filter final csv accordingly
        self.ims_file_path = ims_file_path
        self.spot_id = spot_id
        self.save_dir = save_dir
        self.ims = ImarisDataObject(self.ims_file_path)
        self._configure_instance(spot_id=spot_id)

        del self.ims
        gc.collect()

        # new addition
        self.filename = os.path.basename(ims_file_path).split(".")[0]

    def _configure_instance(self, spot_id: int) -> None:
        """
        Extracts relevant information from ims object and
        instantiates it as instance variables for fast recall.

        Currently Extracts:
            - all the spot names -- List
            - all the stats_names -- {id: pd.DataFrame}
            - all the stats values -- {id: pd.DataFrame}
            - all the factor info -- {id: pd.DataFrame}

        Args:
            spot_id (int): index of the spot ["points0", "points1"]
                spot_id = 0 is information on "points0" etc

        """
        # extract all information and saves it as a instance var
        if spot_id == -1:
            # configure all available spots
            self.spot_names = self.ims.get_object_names("Points")
        else:
            # grab the spot we care about
            self.spot_names = self.ims.get_object_names("Points")
            if (spot_id >= 0) and (spot_id <= len(self.spot_names)):
                self.spot_names = [self.spot_names[spot_id]]
            elif spot_id > len(self.spot_names):
                raise ValueError(
                    f"spot_id {spot_id} exceeds number of spots available {len(self.spot_names)}"
                )
            else:
                # some currently unknown error
                raise NotImplementedError("currently unknown errror lol")

        # get all the stats names for every spot {spot_id: stats_name_df}
        self.stats_names = {
            spot_id: self.ims.get_stats_names(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

        # get all the stats values for every spot {surf_id: stats_values_df}
        self.stats_values = {
            spot_id: self.ims.get_stats_values(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

        # get all the factor table info for every spot {surf_id: factor_df}
        self.factors = {
            spot_id: self.ims.get_object_factor(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

        # gets all the track id information for every spot
        self.track_ids = {
            spot_id: self.ims.get_track_ids(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

        # get all object information for every spot
        self.track_info = {
            spot_id: self.ims.get_track_info(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

        # get all the track object id information for every spot {surf_id: object_ids_series}
        self.object_ids = {
            spot_id: self.ims.get_track_object_ids(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

        # get all object information for every spot {spot_id: object_info_df}
        # TODO: Redundant - get_track_object_info and get_track_object_id is the same
        self.object_info = {
            spot_id: self.ims.get_track_object_info(spot_name)
            for spot_id, spot_name in enumerate(self.spot_names)
        }

    def _organize_stats(self, stats_values: pd.DataFrame) -> Dict:
        """Organized the data such that it looks like
        {ID_Object: {Stats Name: Value}}

        Args:
            spot_stats_values (pd.DataFrame): a single dataframe
            that contains the statistics for a single spot

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

    def _organize_stats2(self, stats_values: pd.DataFrame) -> Dict:
        """Organized the data such that it looks like
        {ID_Object: {Stats Name: Value}}

        Args:
            stats_values (pd.DataFrame): a single dataframe
            that contains the statistics for a single spot

        Returns:
            Dict: _description_
        """
        grouped_stats = {
            obj_id: dict(zip(sub.ID_StatisticsType, sub.Value))
            for obj_id, sub in stats_values.groupby("ID_Object")
        }
        return grouped_stats

    def _format_data(
        self,
        stats_values: Dict,
        stat_names: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates the formatted DataFrame.

        Args:
            stats_values (Dict): nested dictionary of stats values.

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
        spot_id: int,
    ) -> None:
        # a function to write csv information to disk
        # get save_dir/original_filename.csv
        ims_filename = os.path.basename(self.ims_file_path).split(".")[0]
        ims_filename = f"{ims_filename}_spot_wtrack_{(spot_id + 1)}.csv"
        save_filepath = os.path.join(save_dir, ims_filename)
        dataframe.to_csv(save_filepath)

        # store ims_filename
        self.ims_filename = ims_filename

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
        # for spot parser need to filter out track id information
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

        # Keeping this is good for SPOTS
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

                name = "_".join(name.split(" "))  # new
                channel = f"{self.filename}_{name}_Ch={channel}_Img=1"  # new
                channel = f"{name} Channel_{channel}"  # original
                return channel
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
        # TODO: Implement this
        raise NotImplementedError

    def _update_track_id_info(self, spot_name, dataframe) -> pd.DataFrame:
        """
        Creates a new column where for each object ID it indicates which track it belongs to.

        Args:
            spot_name (int): name of the spot ie: "Points0"
            dataframe (pd.DataFrame): formatted dataframe with all the stats.

        Returns:
            pd.DataFrame: dataframe with updated Track_ID column.
        """

        object_ids = self.object_ids.get(spot_name)
        track_info = self.track_info.get(spot_name)

        # temp function to handle the case where when we perform
        # apply and an object id is missing we simply just
        # leave it empty
        def _update_object_with_track_id(database, x):
            try:
                return database[int(x["Object_ID"].item())]
            except KeyError:
                return None

        # create database to make obj to track matching efficient
        # key = numerical object id value = track id the obj belong to
        database = {}
        for idx in range(len(track_info)):
            data = track_info.iloc[idx]
            start = data["IndexTrackObjectBegin"]
            end = data["IndexTrackObjectEnd"]
            track_id = data["ID"]
            for i in range(start, end):
                obj_id = object_ids.iloc[i]
                database[obj_id] = track_id

        func = partial(_update_object_with_track_id, database)
        dataframe["Track_ID"] = dataframe.apply(
            func=func,
            axis=1,
        )

        return dataframe

    def _process(self, spot_id: int) -> None:
        """
        Runs a single end to end parser pipeline on a single spot
        Steps:
            - get stat names for a single spot
            - get stat values for a single spot
            - filter stat values to keep only track ids
            - filter stats values to remove track level stat information
            - rename certian columns (if needed)(need a custom func for this to add channel info)
            - organize the filtered stats
            - generate csv
            - save csv

        Args:
            spot_id (int): _description_
        """
        # gather info for current spot
        spot_name = self.spot_names[spot_id]
        stat_names = self.stats_names.get(spot_id)
        stat_values = self.stats_values.get(spot_id)
        object_id = self.object_ids.get(spot_id)
        factor = self.factors.get(spot_id)

        # update channel and spot names
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)

        # filter stats values by object ids (ie: ignore info related to trackids)
        stat_values = self._filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )

        # organize stats value (most compute used here)
        organized_stats = self._organize_stats2(stat_values)

        # generate csv
        stats_df = self._format_data(organized_stats, stat_names=stat_names)

        # add track id information for each object
        stats_df = self._update_track_id_info(spot_id, stats_df)

        return stats_df

    def get_spot_stats_info(self, spot_id: int) -> List[str]:
        """Returns all the stats information in a given spot id

        Args:
            spot_id (int): _description_

        Returns:
            List[str]: _description_
        """
        # check 1
        if (self.spot_id != -1) and (spot_id != 0):
            raise ValueError(
                f"class is initialized with 1 spot, spot_id should be set to 0 or None"
            )

        # check 2
        if spot_id > len(self.spot_names):
            raise ValueError(
                f"spot_id {spot_id} exceeds number of spots available {len(self.spot_names)}"
            )

        # gather info for current spot
        spot_name = self.spot_names[spot_id]
        stat_names = self.stats_names.get(spot_id)
        stat_values = self.stats_values.get(spot_id)
        object_id = self.object_ids.get(spot_id)
        factor = self.factors.get(spot_id)

        # update channel and spot names
        stat_names = self.update_channel_info(stats_names=stat_names, factor=factor)
        stat_names = self.update_spot_info(stats_names=stat_names, factor=factor)

        # filter stats values by object ids (ie: ignore info related to trackids)
        stat_values = self._filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )

        # create dict that maps stat id to name
        stats_dict = dict(zip(stat_names["ID"], stat_names["Name"]))

        # keep only unique for display
        available_stats_names = [
            stats_dict[ids] for ids in stat_values["ID_StatisticsType"].unique()
        ]

        return stats_dict, available_stats_names

    def extract_and_save(self, spot_id: int, save_dir: str = None) -> None:
        # this function is the funtion that gets called externally
        # we can have this function as a ray method to help with distributed execution
        # check 1
        if (self.spot_id != -1) and (spot_id != 0):
            raise ValueError(
                f"class is initialized with 1 spot, spot_id should be set to 0"
            )

        # check 2
        if spot_id > len(self.spot_names):
            raise ValueError(
                f"spot_id {spot_id} exceeds number of spots available {len(self.spot_names)}"
            )

        # process spot
        dataframe = self._process(spot_id)

        # adjust spot_id based on init mode
        # save spot
        save_dir = save_dir if save_dir else self.save_dir
        if self.spot_id == -1:
            self._save_csv(dataframe, save_dir, spot_id=spot_id)
        else:
            self._save_csv(dataframe, save_dir, spot_id=self.spot_id)

        print(f"[info] -- finished: {self.ims_filename}")

    def inspect(self, spot_id: int) -> Dict:
        """
        Runs a single end to end parser pipeline on a single spot
        and returns all components as a dict.
        Steps:
            - get stat names for a single spot
            - get stat values for a single spot
            - filter stat values to keep only track ids
            - filter stats values to remove track level stat information
            - rename certian columns (if needed)(need a custom func for this to add channel info)
            - organize the filtered stats
            - generate csv
            - save csv

        Args:
            spot_id (int): _description_
        """
        # check 1
        if (self.spot_id != -1) and (spot_id != 0):
            raise ValueError(
                f"class is initialized with 1 spot, spot_id should be set to 0"
            )

        # check 2
        if spot_id > len(self.spot_names):
            raise ValueError(
                f"spot_id {spot_id} exceeds number of spots available {len(self.spot_names)}"
            )

        # dict to hold all values to be returned for inspection
        storage = {}

        # gather info for current spot
        spot_name = self.spot_names[spot_id]
        storage["spot_name"] = spot_name
        stat_names = self.stats_names.get(spot_id)
        storage["stat_names_raw"] = stat_names
        stat_values = self.stats_values.get(spot_id)
        storage["stat_values_raw"] = stat_values
        object_id = self.object_ids.get(spot_id)
        storage["object_id"] = object_id
        factor = self.factors.get(spot_id)
        storage["factor"] = factor

        # update channel and spot names
        stat_names = self._update_channel_info(stats_names=stat_names, factor=factor)
        storage["stat_names_channel_info"] = stat_names

        # filter stats values by object ids (ie: ignore info related to trackids)
        stat_values = self._filter_stats(
            stats_values=stat_values,
            filter_col_names=["ID_Object"],
            filter_values=[object_id],
        )
        storage["stat_values_filtered"] = stat_values

        # organize stats value (most compute used here)
        # organized_stats = self._organize_stats(stat_values)  # old slow version
        # storage["organized_stats"] = organized_stats

        organized_stats = self._organize_stats2(stat_values)  # fast
        storage["organized_stats2"] = organized_stats

        # format the data
        stats_df = self._format_data(organized_stats, stat_names=stat_names)
        storage["stats_df"] = stats_df

        # add track id information for each object
        stats_df = self._update_track_id_info(spot_id, stats_df)
        storage["final_df"] = stats_df

        return storage


#############################################################################
