import gc
import ray
from imaris import ImarisDataObject
from surface_parser import SurfaceParser


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
