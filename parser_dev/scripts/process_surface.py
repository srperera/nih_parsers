import os
import ray
import glob
import numpy as np
from typing import List, Tuple
from utils.utils import get_valid_surfaces, run_ray_actors
from parsers.surface_parser import SurfaceParserDistributed
from imaris.exceptions import NoSurfaceException, NoSurfaceObjectsException

"""
Notes:
"""


#########################################################################################
def run_surface_parser_parallel(
    data_dirs: List[str],
    save_dirs: List[str],
    cpu_cores: int = None,
    surface_ids: Tuple[int] = None,
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
        data_dirs (List[str]): _description_
        save_dirs (List[str]): _description_
        cpu_cores (int, optional): _description_. Defaults to None.
        surface_ids (Tuple[int], optional): _description_. Defaults to None.
    """
    if surface_ids:
        assert isinstance(surface_ids, tuple), "surface_ids must be a tuple"
        assert len(surface_ids) > 0, "surface_ids must not be empty"

    # zip data paths and save dirs
    data_paths = list(zip(data_dirs, save_dirs))

    # summary
    run_summary = {}

    for data_path, save_dir in data_paths:
        actors = []

        # get all imaris files from directory
        imaris_files = glob.glob(os.path.join(os.path.abspath(data_path), "*.ims"))
        run_summary[data_path] = {}

        if len(imaris_files) == 0:
            print(f"[info] -- skipping folder no files found ")
            run_summary[data_path] = "NO FILES IN FOLDER"

        else:
            for file_path in imaris_files:
                # get filename from path
                filename = os.path.splitext(os.path.basename(file_path))[0]
                run_summary[data_path][filename] = {}

                # create dir with same name as filename
                save_path = os.path.join(save_dir, filename)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                # get num of valid surfaces
                # this try/except section is kind of a safty check
                # only working files ie: files with data to parse should have
                # ..actors created.
                try:
                    valid_surface_ids = get_valid_surfaces(data_path=file_path)
                    run_summary[data_path][filename]["all_surfaces"] = valid_surface_ids
                except NoSurfaceException:
                    print(f"[info] -- file {filename} contains no surfaces .. skipping")
                    run_summary[data_path][filename] = "NO SURFACES"
                    continue

                # if surface_ids are provided, filter valid_surface_ids
                if surface_ids:
                    valid_surface_ids = list(
                        filter(lambda x: (x + 1) in surface_ids, valid_surface_ids)
                    )

                if len(valid_surface_ids) == 0:
                    print(f"[info] -- no valid surfaces in {filename} .. skipping file")
                else:
                    print(
                        f"[info] -- creating {len(valid_surface_ids)} actors for {filename}"
                    )
                    # create actors for each surface in current imaris file
                    run_summary[data_path][filename]["extracted_surfaces"] = []
                    for idx in valid_surface_ids:
                        actor = SurfaceParserDistributed.remote(
                            file_path,
                            surface_id=idx,
                            save_dir=save_path,
                        )
                        actors.append(actor)
                        # update summary
                        run_summary[data_path][filename]["extracted_surfaces"].append(
                            idx
                        )
                    print("\n")

        run_summary[data_path]["total surfaces"] = len(actors)

        # generate results
        print(f"[info] -- found {len(actors)} actors")
        print(f"[info] -- extracting data ... ")

        run_ray_actors(actors, cpu_cores)

        print(f"[info] -- complete.")


#########################################################################################
