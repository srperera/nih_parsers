import os
import ray
import glob
import numpy as np
from typing import List
from utils.utils import get_num_surfaces
from parsers.time_step_surface_parser import TimeStepSurfaceParserDistributed
from imaris.exceptions import NoSurfaceException


#########################################################################################
def run_surface_parser_parallel(
    data_paths: List[str],
    save_dir: str = "processed_data",
    cpu_cores: int = None,
) -> None:
    """
    Runs ALL track data in an ims file in parallel.
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
    for data_path in data_paths:
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

                # get num surfaces for file
                num_surfaces = get_num_surfaces(file_path)

                # create actors
                for idx in range(num_surfaces):
                    try:
                        actor = TimeStepSurfaceParserDistributed.remote(
                            file_path, surface_id=idx, save_dir=save_path
                        )
                    except NoSurfaceException:
                        print(
                            f"[info] -- no surface found in {filename}..skipping file"
                        )
                    actors.append(actor)

                print("adding files to actors: ", len(actors))

    # generate results
    # split if too many actors vs cores else run all
    if cpu_cores and cpu_cores < len(actors):
        num_actors = len(actors)
        num_splits = np.round(num_actors / cpu_cores)
        splits = np.array_split(np.asarray(actors, dtype=object), num_splits)
        for split in splits:
            results = ray.get(
                [
                    actor.extract_and_save.remote(surface_id=0)
                    for _, actor in enumerate(split)
                ]
            )
    else:
        results = ray.get(
            [
                actor.extract_and_save.remote(surface_id=0)
                for _, actor in enumerate(actors)
            ]
        )


#########################################################################################
def run_surface_parser_parallel_index(
    data_paths: List[str],
    surface_ids: List[int],
    save_dir: str = "processed_data",
    cpu_cores: int = None,
) -> None:
    """
    Runs SPECIFIED track data in an ims file in parallel.
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
    for data_path in data_paths:
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

                # get num surfaces for file
                num_surfaces = get_num_surfaces(file_path)

                # create actors
                for idx in range(num_surfaces):
                    if (idx + 1) in surface_ids:
                        try:
                            actor = TimeStepSurfaceParserDistributed.remote(
                                file_path, surface_id=idx, save_dir=save_path
                            )
                        except NoSurfaceException:
                            print(
                                f"[info] -- no surface found in {filename}..skipping file"
                            )
                        actors.append(actor)

                print("adding files to actors: ", len(actors))

    # generate results
    if cpu_cores and cpu_cores < len(actors):
        num_actors = len(actors)
        num_splits = np.round(num_actors / cpu_cores)
        splits = np.array_split(np.asarray(actors, dtype=object), num_splits)
        for split in splits:
            results = ray.get(
                [
                    actor.extract_and_save.remote(surface_id=0)
                    for _, actor in enumerate(split)
                ]
            )
    else:
        results = ray.get(
            [
                actor.extract_and_save.remote(surface_id=0)
                for _, actor in enumerate(actors)
            ]
        )


#########################################################################################
