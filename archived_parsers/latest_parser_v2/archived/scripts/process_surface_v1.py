import os
import ray
import glob
import numpy as np
from typing import List
from utils.utils import get_valid_surfaces, run_ray_actors
from parsers.surface_parser import SurfaceParserDistributed
from imaris.exceptions import NoSurfaceException, NoSurfaceObjectsException

"""
Notes:
    available functions:
        * run_surface_parser_parallel
        * run_surface_parser_parallel_index
"""


#########################################################################################
def run_surface_parser_parallel(
    data_dirs: List[str],
    save_dirs: List[str],
    cpu_cores: int = None,
) -> None:
    """
    Runs ALL surfaces in an ims file in parallel.
    Pipeline:
        - For every file in every directory
        - Create a folder a with the same name as the filename inside save_dir provided
        - For every surface inside each file we create a remote actor
        - Once all actors are created we can use the cpu cores provided by user to
            run a fixed chunk of actors in parallel so we dont start too many instances at one
        - If no cpu_cores provided we will run all actors in parallel

    Args:
        data_paths (List[str]): a list of directories
        save_dir (str, optional): dir to save files to. Defaults to "processed_data".
        cpu_cores (int, optional): num of cpu cores allocated for processing. Defaults to None.
    """
    actors = []

    # zip data paths and save dirs
    data_paths = list(zip(data_dirs, save_dirs))

    for data_path, save_dir in data_paths:
        # get all imaris files from directory
        imaris_files = glob.glob(os.path.join(os.path.abspath(data_path), "*.ims"))

        if len(imaris_files) == 0:
            print(f"[info] -- skipping folder no files found ")
            pass

        else:
            for file_path in imaris_files:
                # get filename from path
                filename = os.path.basename(file_path).split(".")[0]

                # create dir with same name as filename
                save_path = os.path.join(save_dir, filename)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                # get num of valid surfaces
                valid_surface_ids = get_valid_surfaces(data_path=file_path)

                # create actors for each surface
                for idx in valid_surface_ids:
                    try:
                        actor = SurfaceParserDistributed.remote(
                            file_path,
                            surface_id=idx,
                            save_dir=save_path,
                        )
                    except NoSurfaceException:
                        print(
                            f"[info] -- no surface found in {filename}..skipping file"
                        )
                    except NoSurfaceObjectsException:
                        print(
                            f"[info] -- surface {idx} in {filename} contains no objects .. skipping file"
                        )
                    actors.append(actor)

                print("adding files to actors: ", len(actors))

    # generate results
    run_ray_actors(actors, cpu_cores)


#########################################################################################
def run_surface_parser_parallel_index(
    data_dirs: List[str],
    surface_ids: List[int],
    save_dirs: List[str],
    cpu_cores: int = None,
) -> None:
    """
    Runs SPECIFIED surfaces in an ims file in parallel.
    Pipeline:
        - For every file in every directory
        - Create a folder a with the same name as the filename inside save_dir provided
        - For every surface specified in surface_ids we create a remote actor
        - Once all actors are created we can use the cpu cores provided by user to
            run a fixed chunk of actors in parallel so we dont start too many instances at one
        - If no cpu_cores provided we will run all actors in parallel.

    Args:
        data_paths (List[str]): _description_
        surface_ids (List[int]): _description_
        save_dir (str, optional): _description_. Defaults to "processed_data".
        cpu_cores (int, optional): _description_. Defaults to None.
    """
    actors = []

    # zip data paths and save dirs
    data_paths = list(zip(data_dirs, save_dirs))

    for data_path, save_dir in data_paths:
        # get all imaris files
        imaris_files = glob.glob(os.path.join(os.path.abspath(data_path), "*.ims"))

        if len(imaris_files) == 0:
            print(f"[info] -- skipping folder no files found ")
            pass

        else:
            for file_path in imaris_files:
                # get filename from path
                filename = os.path.basename(file_path).split(".")[0]

                # create dir with same name as filename
                save_path = os.path.join(save_dir, filename)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                # get num of valid surfaces
                valid_surface_ids = get_valid_surfaces(data_path=file_path)

                # create actors
                for idx in valid_surface_ids:
                    if (idx + 1) in surface_ids:
                        try:
                            actor = SurfaceParserDistributed.remote(
                                file_path,
                                surface_id=idx,
                                save_dir=save_path,
                            )
                        except NoSurfaceException:
                            print(
                                f"[info] -- no surface found in {filename} .. skipping file"
                            )
                        except NoSurfaceObjectsException:
                            print(
                                f"[info] -- surface {idx} in {filename} contains no objects .. skipping file"
                            )
                        actors.append(actor)

                print("adding files to actors: ", len(actors))

    # generate results
    run_ray_actors(actors, cpu_cores)


#########################################################################################
