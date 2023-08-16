from utils import utils
import copy
import ray


########################################################################################
def extract_data(ims_file_path, data, valid_surface, save_path):
    print(f"\n[info] extracting data {ims_file_path} -- surface {valid_surface}")
    try:
        # get surface we want to parse
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

        # check for anomalies in the values
        utils.check_anomalies(surface_stats_names, save_path)

        out = {
            "surface_stats_values": surface_stats_values,
            "surface_stats_names": surface_stats_names,
            "save_path": save_path,
        }
    # except AttributeError:
    #     print(f"skipping file {ims_file_path} -- surface {valid_surface} not found\n")
    #     out = None
    except IndexError:
        print(f"skipping file {ims_file_path} -- surface {valid_surface} not found\n")
        out = None

    return out


########################################################################################
@ray.remote
def process_and_save(
    ims_file_path, surface_stats_values, surface_stats_names, categories_list, save_path
):
    print(f"[info] working on file {ims_file_path}")

    # extract data from original h5py table format into a dict of dicts
    # format = {ID_Object: {ID_StatisticsType: Value}}
    extracted_stats = (
        surface_stats_values.groupby("ID_Object")[["ID_StatisticsType", "Value"]]
        .apply(lambda x: x.set_index("ID_StatisticsType").to_dict(orient="dict"))
        .to_dict()
    )
    extracted_stats = {
        k: v["Value"] for k, v in extracted_stats.items()
    }  # clean up step for line above

    # invert dictionary + name modifications
    # this step is a cosmetic step
    # here key=name, value=num
    inverted_stats_names = utils.invert_stats_dict(surface_stats_names)
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

    print("[info] finished! \n")
