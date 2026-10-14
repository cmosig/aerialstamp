"""AerialStamp: the acquisition date of an aerial (drone or airborne) orthophoto, estimated from the image
and its Sentinel-2 time series as a calibrated distribution over the flight year."""
__version__ = "2.0.0"


def date_orthophoto(*args, **kwargs):
    """See aerialstamp.core.date_orthophoto."""
    from .core import date_orthophoto as _f
    return _f(*args, **kwargs)


__all__ = ["date_orthophoto", "__version__"]
