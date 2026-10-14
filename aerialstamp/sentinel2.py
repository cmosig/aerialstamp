"""Sentinel-2 for one orthophoto: download with sentle (as the training cubes:
7-day mean composites, cloud classification, cloud and snow masks, 10 m in the
local UTM zone), and the per-week area-of-interest series and
orthophoto-to-Sentinel-2 match features (as prep_s2.py of the training code).
"""
import datetime as dt
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom

from .constants import BANDS, MATCH_MAX_SIDE, MATCH_MIN_PIX, MATCH_NAMES, S2_RES, WINDOW_PAD
from .ortho import utm_crs


def grid_for(ortho, region, buffer_m=50.0):
    """10 m grid in the local UTM zone covering `region` (orthophoto CRS) plus a buffer."""
    crs = utm_crs(ortho.lon, ortho.lat)
    l, b, r, t = transform_bounds(ortho.crs, crs, *region)
    l = np.floor((l - buffer_m) / S2_RES) * S2_RES
    b = np.floor((b - buffer_m) / S2_RES) * S2_RES
    r = np.ceil((r + buffer_m) / S2_RES) * S2_RES
    t = np.ceil((t + buffer_m) / S2_RES) * S2_RES
    return crs, (float(l), float(b), float(r), float(t))


def download(zarr_path, crs, bounds, year, start=None, end=None, device="cpu",
             num_workers=4):
    """Weekly Sentinel-2 composites for the submitted year +- WINDOW_PAD days
    (or start/end) into zarr_path with sentle. Returns the zarr path."""
    from sentle import sentle
    start = pd.Timestamp(start) if start else pd.Timestamp(year, 1, 1) - pd.Timedelta(days=WINDOW_PAD + 7)
    end = pd.Timestamp(end) if end else pd.Timestamp(year, 12, 31) + pd.Timedelta(days=WINDOW_PAD + 7)
    end = min(end, pd.Timestamp(dt.date.today()))
    l, b, r, t = bounds
    part = zarr_path + ".part"
    if os.path.exists(part):
        shutil.rmtree(part)
    sentle.process(
        zarr_store=part,
        target_crs=crs,
        target_resolution=S2_RES,
        bound_left=l, bound_bottom=b, bound_right=r, bound_top=t,
        datetime=f"{start.date()}/{end.date()}",
        S2_mask_snow=True,
        S2_cloud_classification=True,
        S2_cloud_classification_device=device,
        S2_apply_cloud_mask=True,
        S2_apply_snow_mask=True,
        S2_nbar=False,
        S1_assets=None,
        time_composite_freq="7d",
        num_workers=num_workers,
        zarr_store_chunk_size=dict(x=100, y=100, time=50),
    )
    os.replace(part, zarr_path)  # only complete downloads are cached
    return zarr_path


def open_cube(path):
    """-> xarray Dataset and its grid dict(crs, transform, shape)."""
    ds = xr.open_zarr(path)
    if "sentle" not in ds:
        raise ValueError(f"{path}: not a sentle cube (no 'sentle' variable)")
    missing = set(BANDS) - set(ds.band.values)
    if missing:
        raise ValueError(f"{path}: Sentinel-2 bands missing: {sorted(missing)}")
    xs, ys = ds.x.values.astype(float), ds.y.values.astype(float)
    wkt = ds.attrs.get("crs_wkt") or ds["spatial_ref"].attrs["crs_wkt"]
    crs = rasterio.crs.CRS.from_wkt(wkt)
    # coordinates are the upper-left corners of the 10 m cells
    tr = rasterio.transform.from_origin(xs.min(), ys.max(), S2_RES, S2_RES)
    return ds, dict(crs=crs, transform=tr, shape=(len(ys), len(xs)))


def aoi_on_grid(ds, grid, aoi=None, aoi_crs=None, gv=None):
    """AOI mask on the cube grid: the cube's forest_label raster where it exists
    (training cubes), else the rasterised AOI, else the orthophoto coverage."""
    H, W = grid["shape"]
    if "forest_label" in ds:
        m = ~np.isnan(ds["forest_label"].values)
        if m.any():
            return m
    if aoi is not None:
        g = transform_geom(aoi_crs, grid["crs"], aoi)
        m = rasterize([g], out_shape=(H, W), transform=grid["transform"], fill=0,
                      default_value=1, dtype=np.uint8).astype(bool)
        if m.any():
            return m
    if gv is not None and (gv > 0.5).any():
        return gv > 0.5
    return np.ones((H, W), bool)


def _corr(a, b):
    """Row-wise Pearson correlation of (T, N) arrays with NaNs masked."""
    m = ~(np.isnan(a) | np.isnan(b))
    n = m.sum(1)
    a0 = np.where(m, a, 0.0)
    b0 = np.where(m, b, 0.0)
    ma = a0.sum(1) / np.maximum(n, 1)
    mb = b0.sum(1) / np.maximum(n, 1)
    da = np.where(m, a - ma[:, None], 0.0)
    db = np.where(m, b - mb[:, None], 0.0)
    num = (da * db).sum(1)
    den = np.sqrt((da**2).sum(1) * (db**2).sum(1))
    r = num / np.where(den > 0, den, np.nan)
    r[n < MATCH_MIN_PIX] = np.nan
    return r


def _rank(x):
    """Ranks along axis 1 ignoring NaN (NaN stay NaN)."""
    out = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        m = ~np.isnan(x[i])
        if m.sum():
            out[i, m] = np.argsort(np.argsort(x[i, m]))
    return out


def series_and_match(ds, aoi, g10=None, gv=None):
    """Per-week AOI series (12 bands, DN 0-10000), clear share and the 12 match
    features. -> dict(time (T,) ns ascending, series (T, 12), clear (T,), match (T, 12))."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (cloudy) weeks
        return _series_and_match(ds, aoi, g10, gv)


def _series_and_match(ds, aoi, g10=None, gv=None):
    rows, cols = np.nonzero(aoi)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    step = int(np.ceil(max(r1 - r0, c1 - c0) / MATCH_MAX_SIDE))
    sl = (slice(r0, r1, step), slice(c0, c1, step))
    aoi = aoi[sl]
    cube = ds["sentle"].sel(band=BANDS).isel(y=sl[0], x=sl[1]).values.astype(np.float32)
    t = ds.time.values
    order = np.argsort(t)
    t, cube = t[order], cube[order]
    T = len(t)
    pix = cube[:, :, aoi]  # (T, 12, N)
    s2clear = ~np.isnan(pix[:, 1:4]).any(1)  # (T, N)
    with np.errstate(invalid="ignore"):
        series = np.nanmean(pix, axis=2).astype(np.float32)
    out = {"time": t.astype("datetime64[ns]").astype(np.int64), "series": series,
           "clear": s2clear.mean(1).astype(np.float32), "n_aoi_px": int(aoi.sum())}
    if g10 is None:
        return out
    g = g10[:, sl[0], sl[1]][:, aoi]
    v = gv[sl[0], sl[1]][aoi]
    good = v > 0.8
    g = np.where(good[None], g, np.nan)
    R, G, B = g
    with np.errstate(invalid="ignore", divide="ignore"):
        obright = (R + G + B) / 3
        ogch = G / (R + G + B)
        orch = R / (R + G + B)
        ogrvi = (G - R) / (G + R)
        b2, b3, b4, b8 = (pix[:, BANDS.index(b)] for b in ("B02", "B03", "B04", "B08"))
        sb = b2 + b3 + b4
        sgch = b3 / sb
        srch = b4 / sb
        sgrvi = (b3 - b4) / (b3 + b4)
        sndvi = (b8 - b4) / (b8 + b4)
        both = s2clear & good[None]

        def rep(x):
            return np.broadcast_to(x, (T, len(x)))

        n_good = max(1, good.sum())
        feats = {
            "n_frac": both.sum(1) / n_good,
            "r_bright": _corr(rep(obright), sb),
            "r_red": _corr(rep(R), b4),
            "r_green": _corr(rep(G), b3),
            "r_blue": _corr(rep(B), b2),
            "r_gchroma": _corr(rep(ogch), sgch),
            "r_grvi": _corr(rep(ogrvi), sgrvi),
            "r_grvi_ndvi": _corr(rep(ogrvi), sndvi),
            "r_bright_nir": _corr(rep(obright), b8),
            "d_gchroma": np.nanmean(np.where(both, sgch, np.nan), 1) - np.nanmean(ogch),
            "d_rchroma": np.nanmean(np.where(both, srch, np.nan), 1) - np.nanmean(orch),
            "r_rank": _corr(_rank(np.where(both, rep(obright), np.nan)),
                            _rank(np.where(both, sb, np.nan))),
        }
    out["match"] = np.stack([feats[k] for k in MATCH_NAMES], 1).astype(np.float32)
    out["n_matched_px_max"] = int(both.sum(1).max()) if T else 0
    return out


def default_cache_path(ortho_path, year, cache_dir=None):
    base = os.path.splitext(os.path.basename(ortho_path))[0]
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(ortho_path)),
                                          ".aerialstamp")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{base}_s2_{year}.zarr")
