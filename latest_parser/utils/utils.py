import gc
import ray
from typing import List
from imaris.imaris import ImarisDataObject
from parsers.surface_parser import SurfaceParser


#########################################################################################
def get_num_surfaces(data_path: str) -> int:
    ims_obj = ImarisDataObject(data_path)
    num_surfaces = len(ims_obj.get_object_names("Surface"))
    return num_surfaces


#########################################################################################
def surface_contains_tracks(
    data_path: str,
    surface_id: int = 0,
) -> bool:
    """Checks ims file to see if track information is avilable for a given surface.

    Args:
        data_path (str): path to ims file
        surface_id (int, optional): surface id value. Defaults to 0.

    Returns:
        bool: Returns True if track information is avilable else False
    """
    ims_obj = ImarisDataObject(data_path)
    surface_names = ims_obj.get_object_names("Surface")
    return ims_obj.contains_tracks(surface_names[surface_id])


#########################################################################################
def contains_sufaces(data_path: str, surface_id: int = 0) -> bool:
    """Checks ims file to see if surface information is avilable.

    Args:
        data_path (str): path to ims file
        surface_id (int, optional): surface id value. Defaults to 0.

    Returns:
        bool: Returns True if track information is avilable else False
    """
    ims_obj = ImarisDataObject(data_path)
    surface_names = ims_obj.get_object_names("Surface")
    return ims_obj.contains_sufaces(surface_names[surface_id])


#########################################################################################
def get_surface_stats(data_path: str, surface_id: int = -1) -> List[str]:
    """_summary_

    Args:
        data_path (str): _description_
        surface_id (int, optional): _description_. Defaults to -1.

    Raises:
        ValueError: _description_

    Returns:
        List[str]: _description_
    """
    num_surfaces = get_num_surfaces(data_path)  # num available surfaces
    parser = SurfaceParser(data_path)
    # grab all
    if surface_id == -1:  # grab all
        available_stats_storage = [
            parser.get_surface_info(surface_id=idx) for idx in range(num_surfaces)
        ]
        return available_stats_storage

    elif surface_id <= num_surfaces:
        available_stats_storage = parser.get_surface_info(surface_id)
        return available_stats_storage

    else:
        raise ValueError("requested surface id exceeeds number of surfaces available")


#########################################################################################
def run_parser_parallel(data_path: str, save_dir: str = None):
    """Runs all the surfaces in an ims file in parallel

    Args:
        data_path (_type_): str path to ims object

    Returns:
        _type_: _description_
    """
    # get number of surfaces in object
    ims_obj = ImarisDataObject(data_path)
    num_surfaces = len(ims_obj.get_object_names("Surface"))

    # run garbage collection and free up memory
    del ims_obj
    gc.collect()

    # run ray
    actors = [SurfaceParser.remote(data_path) for _ in range(num_surfaces)]
    results = ray.get([actor.inspect.remote(idx) for idx, actor in enumerate(actors)])

    return results


#########################################################################################
