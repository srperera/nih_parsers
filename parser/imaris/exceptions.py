############################################
class NoTrackException(Exception):
    """raised when no Tracks are Found

    Args:
        Exception (_type_): _description_
    """

    pass


############################################
class NoSurfaceException(Exception):
    """raised when no Surfaces are Found

    Args:
        Exception (_type_): _description_
    """

    pass


###########################################
class NoFirstObjectException(Exception):
    """raised when no NoFirstObject are Found

    Args:
        Exception (_type_): _description_
    """

    pass


###########################################
class NoSurfaceObjectsException(Exception):
    """raised when no objects in a given surface are Found

    Args:
        Exception (_type_): _description_
    """

    pass
