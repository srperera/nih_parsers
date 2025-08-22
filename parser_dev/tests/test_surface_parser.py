import unittest
import ray
import pandas as pd
from imaris.imaris import ImarisDataObject
from parsers.surface_parser import SurfaceParserDistributed
from typing import List, Dict
from tqdm import tqdm


###########################################################################
class TestSurfaceParser(unittest.TestCase):
    def setUp(self):
        parser_ims_path = "C:/Users/perer/Downloads/FIX/FIX/04052023 SHIV aLN iLN 6-color/04052023 SHIV aLN iLN 6-color/ILN P3 whole tiled 2.ims"
        test_surface_id = 0
        parser1 = SurfaceParserDistributed.remote(
            parser_ims_path, surface_id=test_surface_id
        )
        parser2 = SurfaceParserDistributed.remote(
            parser_ims_path, surface_id=test_surface_id
        )

        # get generated df
        final_df = parser1.inspect.remote(surface_id=0)  # must be 0
        final_df = ray.get(final_df)
        self.generated_df = final_df["stats_df"]

        # get stats dict that maps ID to Name
        stats_dict = parser2.get_surface_stats_info.remote(surface_id=0)  # must be 0
        self.stats_dict = ray.get(stats_dict)

        # get original data
        ims_obj = ImarisDataObject(parser_ims_path)
        self.raw_stats = ims_obj.get_stats_values(final_df["surface_name"])

    def validate(
        self,
        gen_stats: pd.DataFrame,
        raw_obj_stats: pd.DataFrame,
        stats_dict: Dict,
    ) -> None:
        """
        Given a single row of statistics ie: all the stats for a single
        objectid and the raw stats values dictionary for the same object id
        this function will compare each statistics value between the generated
        version and the raw stats values dictionary obtained directly from the
        imaris file to ensure every stat values match for the given object id.

        Args:
            gen_stats (pd.DataFrame): dataframe for all the stats for a single object id
            raw_obj_stats (pd.DataFrame): raw stats values for a single object id
            stats_dict (Dict): dictionary that maps Stats ID Number to Stat Name

        Raises:
            ValueError: _description_
        """
        raw_obj_id = raw_obj_stats["ID_Object"].unique().item()
        gen_obj_id = gen_stats["Object_ID"].item()
        assert raw_obj_id == gen_obj_id, "Object IDs Do NOT Match"

        gen_values = {}
        # key = stats name, value=generated stats value for stats name
        for stats_name in gen_stats.columns.to_list():
            gen_values[stats_name] = gen_stats[stats_name].item()
        gen_values.pop("Object_ID")

        raw_values = {}
        for stat_id in raw_obj_stats["ID_StatisticsType"].to_list():
            stats_name = stats_dict[stat_id]
            stats_value = raw_obj_stats[raw_obj_stats["ID_StatisticsType"] == stat_id]
            raw_values[stats_name] = stats_value["Value"].item()

        # double check all keys in both dicts are the same
        assert list(gen_values.keys()) == list(raw_values.keys())

        for key in gen_values.keys():
            raw_value = raw_values[key]
            gen_value = gen_values[key]
            # if raw_value != gen_value:
            #     print(key, raw_value, gen_value)
            #     raise ValueError
            self.assertEqual(raw_value, gen_value)

        print(f"[info] -- confirmed all values for object {raw_obj_id}")

    def test_stats_validation(self) -> None:
        """_summary_

        Args:
            generated_df (pd.DataFrame): _description_
            raw_stats (pd.DataFrame): _description_
            stats_dict (Dict): _description_
        """
        for object_id in tqdm(self.generated_df["Object_ID"].to_list()):
            # get generated stats info for object id
            gen_stats = self.generated_df[self.generated_df["Object_ID"] == object_id]
            # get raw stats values for object id
            raw_obj_stats = self.raw_stats[self.raw_stats["ID_Object"] == object_id]

            self.validate(gen_stats, raw_obj_stats, self.stats_dict)


if __name__ == "__main__":
    unittest.main()
