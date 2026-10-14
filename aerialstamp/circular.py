"""Day-of-year helpers on the fixed 365-bin circle.

Bin b in [0, 365) covers the fraction [b/365, (b+1)/365) of the calendar year,
so leap years are squeezed by one day rather than given a 366th class.
"""
import datetime as dt

import numpy as np

from .constants import NBINS

DAYS = np.arange(NBINS)
_D = None


def days_in_year(year):
    year = np.asarray(year)
    return np.where((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), 366, 365)


def doy_to_bin(doy, year):
    """1-based day of year -> continuous bin position in [0, 365)."""
    return (np.asarray(doy, float) - 1.0) * NBINS / days_in_year(year)


def date_to_bin(date):
    date = to_date(date)
    return float(doy_to_bin(date.timetuple().tm_yday, date.year))


def bin_to_date(b, year):
    diy = int(days_in_year(year))
    doy = int(np.floor(b * diy / NBINS)) + 1
    return dt.date(int(year), 1, 1) + dt.timedelta(days=doy - 1)


def to_date(d):
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


def circ_diff(a, b):
    """Signed circular difference a - b in bins, in (-182.5, 182.5]."""
    d = (np.asarray(a, float) - np.asarray(b, float)) % NBINS
    return np.where(d > NBINS / 2, d - NBINS, d)


def wrapped_gaussian(center, sigma):
    """(len(center), 365) rows of a wrapped Gaussian, each summing to 1."""
    center = np.atleast_1d(np.asarray(center, float))
    d = circ_diff(DAYS[None, :] + 0.5, center[:, None])
    k = np.exp(-0.5 * (d / sigma)**2)
    return k / k.sum(1, keepdims=True)


def hdi_mask(p, mass):
    """(N, 365) mask of the highest-density set (prediction interval) holding `mass`."""
    p = np.atleast_2d(p)
    order = np.argsort(-p, axis=1)
    cs = np.cumsum(np.take_along_axis(p, order, 1), axis=1)
    k = (cs < mass).sum(1) + 1
    mask = np.zeros_like(p, bool)
    rows = np.arange(len(p))[:, None]
    mask[rows, order] = np.arange(NBINS)[None, :] < k[:, None]
    return mask


def mask_to_arcs(mask):
    """Contiguous circular periods [(start_bin, end_bin_inclusive), ...] of a mask."""
    if mask.all():
        return [(0, NBINS - 1)]
    if not mask.any():
        return []
    off = int(np.argmin(mask))
    m = np.roll(mask, -off)
    arcs, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if not v and start is not None:
            arcs.append(((start + off) % NBINS, (i - 1 + off) % NBINS))
            start = None
    if start is not None:
        arcs.append(((start + off) % NBINS, (NBINS - 1 + off) % NBINS))
    return arcs


def arc_bins(a, b):
    return np.arange(a, b + 1) if a <= b else np.r_[np.arange(a, NBINS), np.arange(0, b + 1)]


def circular_median(p):
    """Bin minimising the expected absolute circular error (the point estimate)."""
    global _D
    if _D is None:
        _D = np.abs(circ_diff(DAYS[:, None], DAYS[None, :]))
    return (np.atleast_2d(p) @ _D).argmin(1)
