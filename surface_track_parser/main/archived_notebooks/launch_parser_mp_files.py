import sys

sys.path.append(".")

import os
import gc
import glob
import time
import argparse
from utils import utils
import concurrent.futures
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # disables read filelock


############################################################################################
def generate_statistics(config_path: str):
    # load config path
    yaml = utils.load_yaml(config_path)

    # type of parser
    parser_type = yaml["parser_type"]

    if parser_type == "track":
        from parsers.track_parser import extract_data, process_and_save

        print(f"[info] -- parser mode: {parser_type}")

    elif parser_type == "surface":
        from parsers.surface_parser import extract_data, process_and_save

        print(f"[info] -- parser mode: {parser_type}")

    elif parser_type == "first":
        from parsers.first_surface_parser import extract_data, process_and_save

        print(f"[info] -- parser mode: {parser_type}")

    else:
        raise ValueError("Invalid Parser Type")

    # files to scan
    directories = yaml["data_dir"]

    # get the stats categories
    stats_categories = utils.read_txt(yaml["stats_category_path"])

    # valid surface
    valid_surfaces = yaml["valid_surface"]

    # iterate, process and save
    # parallel processes all files within 1 directory at at time
    for idx, directory in tqdm(enumerate(directories)):
        print(f"[info] -- processing directory index: {idx+1}/{len(directories)}")

        # save dir
        save_dir = yaml["save_dir"][idx]

        # grab all the files in the directory w/ .ims
        filenames = list(glob.glob(os.path.join(directory, "*.ims")))

        # create a list to hold ray subprocess
        processes = []
        for filename in filenames:
            print(f"\n[info] -- processing file {os.path.basename(filename)}\n")

            # load the imaris file
            data = utils.load_ims(filename)

            for surface in valid_surfaces:
                # create folder
                folder_path = os.path.join(save_dir, str(surface))
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)

                # convert to zero indexed surface value
                current_surface = int(surface) - 1

                # save_file_path
                save_path = utils.get_save_filepath(
                    parser_type, folder_path, filename, (current_surface + 1)
                )

                # extract data
                extracted_data = extract_data(
                    filename, data, current_surface, save_path
                )

                # process and save
                if extracted_data:
                    # append stats categories to extracted data
                    extracted_data.update({"categories_list": stats_categories})
                    processes.append(extracted_data)

                else:
                    # there is no surface deleting folder
                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)

        print("\n\t[info] -- finished data extraction")
        print(f"\t[info] -- found {len(processes)} surfaces")
        print(f"\t[info] -- processing {len(processes)} surfaces\n")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            executor.map(process_and_save, processes)

        # memory clean up
        del data
        del processes
        gc.collect()

    print(f"\n[info] -- DONE")


############################################################################################
if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description="Launch Imaris Parser")
    parser.add_argument(
        "--config",
        help="path to config file. example: 'configs/config.yaml'",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    start = time.perf_counter()
    generate_statistics(config_path=args.config)
    stop = time.perf_counter()
    print(f"Total Run Time: {stop - start}")
