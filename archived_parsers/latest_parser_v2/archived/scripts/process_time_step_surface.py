import os
import ray
import glob
import numpy as np
from typing import List
from utils.utils import get_valid_track_surfaces
from imaris.exceptions import NoSurfaceException, NoTrackException
from parsers.time_step_surface_parser import TimeStepSurfaceParserDistributed


#########################################################################################
def run_surface_timestep_parser_parallel(
    data_dirs: List[str],
    save_dirs: List[str],
    cpu_cores: int = None,
    time_step: float = 1.0,
) -> None:
    """
    Runs ALL available surfaces in an ims file in parallel.
    Pipeline:
        - For every file in every directory
        - Create a folder a with the same name as the filename inside save_dir provided
        - For every surface specified in surface_ids we create a remote actor
        - Once all actors are created we can use the cpu cores provided by user to
            run a fixed chunk of actors in parallel so we dont start too many instances at one
        - If no cpu_cores provided we will run all actors in parallel.

    Args:
        data_dirs (List[str]): _description_
        save_dirs (List[str]): _description_
        cpu_cores (int, optional): _description_. Defaults to None.
        time_step (float, optional): _description_. Defaults to 1.0.
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
                valid_surface_ids = get_valid_track_surfaces(data_path=file_path)

                # create actors
                for idx in valid_surface_ids:
                    try:
                        actor = TimeStepSurfaceParserDistributed.remote(
                            file_path,
                            surface_id=idx,
                            save_dir=save_path,
                            time_step=time_step,
                        )
                    except NoSurfaceException:
                        print(
                            f"[info] -- no surface found in {filename} .. skipping file"
                        )
                    except NoTrackException:
                        print(
                            f"[info] -- no tracks found in {filename} .. skipping file"
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
def run_surface_timestep_parser_parallel_index(
    data_dirs: List[str],
    save_dirs: List[str],
    surface_ids: List[int],
    cpu_cores: int = None,
    time_step: float = 1.0,
) -> None:
    """
    Runs SPECIFIED surfaces if available in an ims file in parallel.
    Pipeline:
        - For every file in every directory
        - Create a folder a with the same name as the filename inside save_dir provided
        - For every surface specified in surface_ids we create a remote actor
        - Once all actors are created we can use the cpu cores provided by user to
            run a fixed chunk of actors in parallel so we dont start too many instances at one
        - If no cpu_cores provided we will run all actors in parallel.

    Args:
        data_dirs (List[str]): _description_
        save_dirs (List[str]): _description_
        surface_ids (List[int]): _description_
        cpu_cores (int, optional): _description_. Defaults to None.
        time_step (float, optional): _description_. Defaults to 1.0.
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
                valid_surface_ids = get_valid_track_surfaces(data_path=file_path)

                # create actors
                for idx in valid_surface_ids:
                    if (idx + 1) in surface_ids:
                        try:
                            actor = TimeStepSurfaceParserDistributed.remote(
                                file_path,
                                surface_id=idx,
                                save_dir=save_path,
                                time_step=time_step,
                            )
                        except NoSurfaceException:
                            print(
                                f"[info] -- no surface found in {filename} .. skipping file"
                            )
                        except NoTrackException:
                            print(
                                f"[info] -- no tracks found in {filename} .. skipping file"
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
