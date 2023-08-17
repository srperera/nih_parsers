from utils import utils, exceptions
import os
import gc


########################################################################################
def extract_data(ims_file_path, data, valid_surface, save_path):
    print(
        f"\t[info] extracting data: {os.path.basename(ims_file_path)} -- surface: {valid_surface}"
    )

    try:
        surface_name = utils.get_object_names(
            full_data_file=data, search_for="Surface"
        )[valid_surface]

        # get all the statistics names
        surface_stats_names = utils.get_statistics_names(
            full_data_file=data, object_name=surface_name
        )

        # get the statistics values in the surface
        surface_stats_values = utils.get_stats_values(
            full_data_file=data, object_name=surface_name
        )

        # get only the surface ids that are in the first time point
        track_ids = utils.get_first_timeframe_objects(
            stats_values=surface_stats_values, stats_dict=surface_stats_names
        )

        # filter full stats values based on track ids
        filtered_stats_df = utils.filter_by_object_id(
            stats_values=surface_stats_values, object_ids=track_ids
        )

        # filter stats dict
        filtered_dict = utils.filter_stats_dict(
            stats_dict=surface_stats_names, filtered_df=filtered_stats_df
        )

        # check for anomalies in the values
        utils.check_anomalies(surface_stats_names, save_path)

        extracted_data = {
            "ims_file_path": ims_file_path,
            "filtered_stats_df": filtered_stats_df,
            "filtered_dict": filtered_dict,
            "save_path": save_path,
            "valid_surface": valid_surface,
        }
    except exceptions.NoSurfaceException:
        print("\t\t[info] -- raised no surface exception")
        print(
            f"\t\t[info] -- skipping file {os.path.basename(ims_file_path)} -- no surfaces found"
        )
        extracted_data = None
    except IndexError:
        print("\t\t[info] -- raised index error")
        print(
            f"\t\t[info] -- skipping file {os.path.basename(ims_file_path)} -- surface {valid_surface} not found"
        )
        extracted_data = None
    except exceptions.NoTrackException:
        print("\t\t[info] -- raised no track exception")
        print(
            f"\t\t[info] -- skipping file {os.path.basename(ims_file_path)} -- tracks not found in surface: {valid_surface}"
        )
        extracted_data = None
    return extracted_data


########################################################################################
def process_and_save(extracted_data):
    try:
        ims_file_path = extracted_data.get("ims_file_path")
        extracted_stats = extracted_data.get("filtered_stats_df")
        filtered_dict = extracted_data.get("filtered_dict")
        save_path = extracted_data.get("save_path")
        categories_list = extracted_data.get("categories_list")
        valid_surface = extracted_data.get("valid_surface")

        print(
            f"\t[info] working on file: {os.path.basename(ims_file_path)} -- surface {valid_surface}"
        )

        # extract data from original h5py table format into a dict of dicts
        # format = {ID_Object: {ID_StatisticsType: Value}}
        extracted_stats = (
            extracted_stats.groupby("ID_Object")[["ID_StatisticsType", "Value"]]
            .apply(lambda x: x.set_index("ID_StatisticsType").to_dict(orient="dict"))
            .to_dict()
        )
        extracted_stats = {
            k: v["Value"] for k, v in extracted_stats.items()
        }  # clean up step for line above

        # invert dictionary + name modifications
        # this step is a cosmetic step
        # here key=name, value=num
        inverted_stats_names = utils.invert_stats_dict(filtered_dict)
        inverted_stats_names = utils.flatten(inverted_stats_names)

        # create a list of stats names (in integer form) we want to remove
        del_list = utils.create_del_list(inverted_stats_names, categories_list)

        # reverse the stats names again such that key=num, value=name
        final_stats_names = {v: k for k, v in inverted_stats_names.items()}

        # filter the user defined category list
        # if the user requested a category that is NOT in the data, remove it
        categories_list_updated = []
        for category in categories_list:
            if category not in final_stats_names.values():
                pass
            else:
                categories_list_updated.append(category)

        categories_list = categories_list_updated
        categories_list.insert(0, "ID")

        # generate csv
        dataframe = utils.generate_csv(
            data_dict=extracted_stats,
            del_list=del_list,
            stats_names=final_stats_names,
            categories_list=categories_list,
        )

        # Addition 04/24/2023 drop the first row with ID -1
        dataframe = dataframe.iloc[1:]

        # save the dataframe in the same location
        dataframe.to_csv(save_path, index=None)

        print(
            f"\t[info] finished: {os.path.basename(ims_file_path)} -- surface {valid_surface} -- processed {len(dataframe)} items"
        )

        del dataframe
        del extracted_data
        del extracted_stats
        gc.collect()

    except Exception as e:
        print(
            f"\tERROR in FILE: {os.path.basename(ims_file_path)} - surface {valid_surface} raised Exception [{e}]"
        )
