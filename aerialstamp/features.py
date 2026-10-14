"""Model inputs: Sentinel-2 week tokens of the flight-year window and the site
facts (as assemble.py of the training code)."""
import numpy as np
import pandas as pd

from .circular import days_in_year
from .constants import BANDS, MATCH_NAMES, N_WEEKS, NBINS, STATIC_MASKED, WINDOW_PAD


def week_features(series, clear, match):
    """(T, 32) week features: 12 sqrt bands, NDVI, NDMI, GRVI, green and red
    chromaticity (17), clear share, S2 flag (2), 12 match features and their flag (13)."""
    s2 = series[:, :12] / 10000.0
    b = {k: s2[:, i] for i, k in enumerate(BANDS[:12])}
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"])
        ndmi = (b["B08"] - b["B11"]) / (b["B08"] + b["B11"])
        grvi = (b["B03"] - b["B04"]) / (b["B03"] + b["B04"])
        vis = b["B02"] + b["B03"] + b["B04"]
        gch = b["B03"] / vis
        rch = b["B04"] / vis
    s2ok = ~np.isnan(s2).any(1)
    opt = np.concatenate([np.sqrt(np.clip(s2, 0, None)), ndvi[:, None], ndmi[:, None],
                          grvi[:, None], gch[:, None] * 3 - 1, rch[:, None] * 3 - 1], 1)
    opt = np.where(s2ok[:, None], np.nan_to_num(opt), 0.0)
    if match is None:
        m = np.zeros((len(series), len(MATCH_NAMES)))
        mok = np.zeros(len(series), bool)
    else:
        mok = (~np.isnan(match[:, 1])) & (match[:, 0] > 0.3)
        m = np.where(mok[:, None], np.nan_to_num(match), 0.0)
    out = np.concatenate([opt, clear[:, None], s2ok[:, None], m, mok[:, None]], 1)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)


def s2_window(z, year):
    """Week tokens of the flight year +- WINDOW_PAD days.
    -> dict(f (k, 32), pos (k,) in bins from 1 January, has_match) or {}."""
    t = pd.DatetimeIndex(z["time"])
    dd = (t - pd.Timestamp(year, 1, 1)).total_seconds().values / 86400.0
    diy = float(days_in_year(year))
    sel = (dd >= -WINDOW_PAD) & (dd < diy + WINDOW_PAD)
    if not sel.sum():
        return {}
    match = z.get("match")
    f = week_features(z["series"], z["clear"], match)[sel][:N_WEEKS]
    return {"f": f, "pos": dd[sel][:N_WEEKS] * NBINS / diy,
            "has_match": match is not None and bool(f[:, -1].any())}


def week_arrays(w):
    """Padded (N_WEEKS, 32) tokens, positions and mask from s2_window output."""
    s2w = np.zeros((N_WEEKS, 32), np.float32)
    s2pos, s2m = np.zeros(N_WEEKS, np.float32), np.zeros(N_WEEKS, bool)
    if w:
        k = len(w["f"])
        s2w[:k], s2pos[:k], s2m[:k] = w["f"], w["pos"], True
    return s2w, s2pos, s2m


def static_features(lat, lon, year, gsd):
    """(10,) site facts: sin/cos of latitude and longitude, latitude, hemisphere,
    year, two masked availability flags, log pixel size."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    s = np.array([np.sin(la), np.cos(la), np.sin(lo), np.cos(lo), lat / 90.0, np.sign(lat),
                  (year - 2021) / 4.0, 0.0, 0.0, np.log10(np.float32(gsd)) + 1.5], np.float32)
    s[list(STATIC_MASKED)] = 0.0
    return s
