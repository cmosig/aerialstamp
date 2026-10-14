"""date_orthophoto: the whole pipeline for one orthophoto."""
import datetime as dt
import os
import warnings

import numpy as np

from . import __version__
from .circular import bin_to_date, to_date
from .constants import N_PATCHES
from .features import s2_window, static_features, week_arrays
from .model import load_backbone, load_ensemble, predict
from .ortho import Orthophoto, footprint, load_aoi, on_grid, sample_patches
from .sentinel2 import (aoi_on_grid, default_cache_path, download, grid_for, open_cube,
                        series_and_match)
from .summary import calibrate, summarise


def date_orthophoto(orthophoto, year=None, date=None, aoi=None, mode="auto", cube=None,
                    download_s2=True, cache_dir=None, s2_start=None, s2_end=None,
                    device="cpu", seed=0, num_workers=4, lat=None, lon=None,
                    return_inputs=False):
    """Estimate the acquisition date of a drone orthophoto.

    orthophoto: path to an RGB GeoTIFF (any projected or geographic CRS).
    year: the flight year (required unless `date` is given); the model dates the
        flight within this year.
    date: a submitted date to check (YYYY-MM-DD); adds plausibility and flag.
    aoi: optional vector file (GeoPackage, GeoJSON, Shapefile) with the area of
        interest, e.g. the part of the orthophoto without reconstruction artefacts.
    mode: "auto" (S2 model when Sentinel-2 weeks with match features exist in the
        window, else the image-only model), "s2" or "image".
    cube: an existing sentle cube (zarr) to use instead of downloading.
    download_s2: download Sentinel-2 with sentle when no cube is given.
    cache_dir: where downloaded cubes are kept (default: .aerialstamp/ next to the orthophoto).
    s2_start, s2_end: override the download window (default: year +- 42 days).
    device: "cpu" or "cuda" (DINOv2 embedding, model, sentle cloud classifier).
    seed: seed of the random patch positions.
    lat, lon: site coordinates (default: centre of the orthophoto).

    Returns a dict (JSON-serialisable) with the predicted date, prediction intervals,
    daily probabilities, the check of `date` and provenance.
    """
    if date is not None:
        date = to_date(date)
        year = year or date.year
        if date.year != year:
            raise ValueError("date and year disagree")
    if year is None:
        raise ValueError("give the flight year (or a date to check)")
    year = int(year)
    if mode not in ("auto", "s2", "image"):
        raise ValueError("mode must be auto, s2 or image")
    notes = []
    with Orthophoto(orthophoto, lat=lat, lon=lon) as ortho:
        aoi_geom = load_aoi(aoi, ortho.crs) if isinstance(aoi, (str, os.PathLike)) else aoi
        week, s2info = None, {}
        if mode in ("auto", "s2"):
            try:
                # download first: sentle forks worker processes, which cannot use
                # CUDA once the parent process has initialised it
                if cube is None:
                    if not download_s2:
                        raise RuntimeError("no cube given and download disabled")
                    cube = default_cache_path(orthophoto, year, cache_dir)
                    if not os.path.exists(cube):
                        crs, bounds = grid_for(ortho, footprint(ortho, aoi_geom))
                        download(cube, crs, bounds, year, s2_start, s2_end, device=device,
                                 num_workers=num_workers)
                ds, grid = open_cube(cube)
                g10, gv = on_grid(ortho, aoi_geom, grid)
                if not (gv > 0.8).any():
                    raise RuntimeError("the orthophoto does not cover any 10 m cell of the cube")
                aoi_grid = aoi_on_grid(ds, grid, aoi_geom, ortho.crs, gv)
                z = series_and_match(ds, aoi_grid, g10, gv)
                w = s2_window(z, year)
                n_match = int(w["f"][:, -1].sum()) if w else 0
                s2info = dict(cube=os.path.abspath(str(cube)), weeks_in_window=len(w["f"]) if w else 0,
                              weeks_with_match_features=n_match,
                              weeks_mostly_clear=int((w["f"][:, 17] >= 0.5).sum()) if w else 0,
                              last_week=str(dt.datetime.utcfromtimestamp(z["time"].max() / 1e9).date())
                              if len(z["time"]) else None,
                              aoi_pixels_10m=int(aoi_grid.sum()))
                if w and w["has_match"]:
                    week = week_arrays(w)
                else:
                    raise RuntimeError("no Sentinel-2 week of the window has match features")
            except Exception as e:  # noqa: BLE001
                if mode == "s2":
                    raise
                notes.append(f"Sentinel-2 not used ({e}); fell back to the image-only model")
                warnings.warn(notes[-1])
        patches, frac = sample_patches(ortho, aoi_geom, seed=seed)
        emb = load_backbone(device)(patches).astype(np.float16).astype(np.float32)
        valid = np.ones(N_PATCHES, bool)
        static = static_features(ortho.lat, ortho.lon, year, ortho.gsd)
        prov = dict(aerialstamp=__version__, orthophoto=os.path.abspath(str(orthophoto)),
                    year=year, lat=round(ortho.lat, 6), lon=round(ortho.lon, 6),
                    gsd_cm=round(ortho.gsd * 100, 2), aoi=str(aoi) if aoi else None,
                    patches=int(len(patches)), patch_valid_share=round(float(frac.mean()), 3),
                    seed=seed)
        kind = "s2" if week is not None else "image"
        models, cal = load_ensemble(kind, device)
        if kind == "s2":
            p_raw = predict(models, static, emb, valid, *week, device=device)
        else:
            p_raw = predict(models, static, emb, valid, device=device)
        p = calibrate(p_raw, cal["T"], cal["s"])
    out = dict(model="S2 model" if kind == "s2" else "image-only model", mode=mode,
               **summarise(p, year, date))
    out["daily_probabilities"] = {str(bin_to_date(b, year)): round(float(v), 6)
                                  for b, v in enumerate(p)}
    out["provenance"] = dict(prov, sentinel2=s2info or None, calibration=cal, notes=notes)
    if return_inputs:
        out["_inputs"] = dict(patches=patches, emb=emb, static=static, week=week, p=p,
                              p_raw=p_raw)
    return out
