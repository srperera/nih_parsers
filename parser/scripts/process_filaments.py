import os
import glob
from typing import List, Tuple
from imaris.exceptions import NoFilamentsException
from utils.utils import get_valid_filaments, run_ray_filament_actors
from parsers.filament_parser import FilamentParserDistributed

"""
Notes:
"""


#########################################################################################
def run_filament_parser_parallel(
    data_dirs: List[str],
    save_dirs: List[str],
    cpu_cores: int = None,
    filament_ids: Tuple[int] = None,
) -> None:
    """
    Runs ALL filaments in an ims file in parallel.
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
        filament_ids (Tuple[int], optional): _description_. Defaults to None.
    """
    if filament_ids:
        assert isinstance(filament_ids, tuple), "filament_ids must be a tuple"
        assert len(filament_ids) > 0, "filament_ids must not be empty"

    # zip data paths and save dirs
    data_paths = list(zip(data_dirs, save_dirs))

    for data_path, save_dir in data_paths:
        actors = []

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

                # get num of valid filaments
                # this try/except section is kind of a safty check
                # only working files ie: files with data to parse should have
                # ..actors created.
                try:
                    valid_filaments_ids = get_valid_filaments(data_path=file_path)
                except NoFilamentsException:
                    print(
                        f"[info] -- file {filename} contains no filaments .. skipping"
                    )
                    break

                # if filament_ids are provided, filter valid_filaments_ids
                if filament_ids:
                    valid_filaments_ids = list(
                        filter(lambda x: (x + 1) in filament_ids, valid_filaments_ids)
                    )

                if len(valid_filaments_ids) == 0:
                    print(
                        f"[info] -- no valid filaments in {filename} .. skipping file"
                    )
                else:
                    print(
                        f"[info] -- creating {len(valid_filaments_ids)} actors for {filename}"
                    )
                    # create actors for each surface in current imaris file
                    for idx in valid_filaments_ids:
                        actor = FilamentParserDistributed.remote(
                            file_path,
                            filament_id=idx,
                            save_dir=save_path,
                        )
                        actors.append(actor)
                    print("\n")

        # generate results
        print(f"[info] -- found {len(actors)} actors")
        print(f"[info] -- extracting data ... ")

        run_ray_filament_actors(actors, cpu_cores)

        print(f"[info] -- complete.")


#########################################################################################
