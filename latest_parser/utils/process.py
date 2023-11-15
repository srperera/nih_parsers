import ray
from typing import List
from .utils import get_num_surfaces
from parsers.surface_parser import SurfaceParserDistributed


#########################################################################################
def run_surface_parser_parallel(data_path: str, save_dir: str = None):
    """Runs all the surfaces in an ims file in parallel

    Args:
        data_path (_type_): str path to .ims object

    Returns:
        _type_: _description_
    """
    # get number of surfaces in object
    num_surfaces = get_num_surfaces(data_path)

    # run ray
    actors = [
        SurfaceParserDistributed.remote(data_path, surface_id=idx)
        for idx in range(num_surfaces)
    ]
    # results = ray.get(
    #     [actor.inspect.remote(surface_id=0) for _, actor in enumerate(actors)]
    # )
    results = ray.get(
        [
            actor.extract_and_save.remote(surface_id=0, save_dir=save_dir)
            for _, actor in enumerate(actors)
        ]
    )

    return results
