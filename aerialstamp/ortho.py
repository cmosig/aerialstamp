"""Reading the orthophoto: area of interest, 10 cm patches, and the orthophoto
averaged onto the 10 m Sentinel-2 grid (as prep_ortho.py of the training code).

All reads go through the orthophoto's overviews where they exist, so a 10 cm
patch from a 2 cm orthophoto reads the 8 cm overview, not full-resolution tiles.
"""
import numpy as np
import rasterio
import rasterio.warp
import shapely
import shapely.geometry
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds, transform_geom
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter

from .constants import MASK_MAX_PX, N_PATCHES, PATCH_GSD, PATCH_PX


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return rasterio.crs.CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def ground_scale(crs, lat):
    """CRS units per ground metre: 1/cos(lat) for Web Mercator (the
    deadtrees.earth COGs), 1 for other metric projected CRSs."""
    if crs.to_epsg() in (3857, 900913, 3395) or "Mercator" in (crs.to_wkt() or "") and \
            "Transverse" not in crs.to_wkt():
        return 1.0 / np.cos(np.deg2rad(lat))
    units = (crs.linear_units or "").lower()
    if units not in ("metre", "meter", "m", "metres", "meters"):
        raise ValueError(f"orthophoto CRS {crs} has linear units '{crs.linear_units}'; "
                         "expected metres")
    return 1.0


class Orthophoto:
    """An orthophoto opened for reading; geographic CRSs are warped to UTM."""

    def __init__(self, path, lat=None, lon=None):
        """lat/lon: site coordinates (default: centre of the orthophoto); they set
        the Web Mercator scale and the site facts of the model."""
        self.path = path
        self._raw = rasterio.open(path)
        assert self._raw.count >= 3, f"{path}: need an RGB orthophoto (>= 3 bands)"
        L, B, R, T = self._raw.bounds
        lon_c, lat_c = rasterio.warp.transform(self._raw.crs, "EPSG:4326",
                                               [(L + R) / 2], [(B + T) / 2])
        self.lon = float(lon_c[0]) if lon is None else float(lon)
        self.lat = float(lat_c[0]) if lat is None else float(lat)
        if self._raw.crs.is_geographic:
            self.src = WarpedVRT(self._raw, crs=utm_crs(self.lon, self.lat),
                                 resampling=Resampling.bilinear)
        else:
            self.src = self._raw
        self.crs = self.src.crs
        self.scale = ground_scale(self.crs, self.lat)
        self.gsd = float(self.src.res[0] / self.scale)

    def close(self):
        if self.src is not self._raw:
            self.src.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def load_aoi(path, crs):
    """Union of all polygons of a vector file, as a GeoJSON-like mapping in `crs`."""
    import pyogrio
    df = pyogrio.read_dataframe(path)
    geom = shapely.union_all(df.geometry.values)
    src_crs = df.crs or "EPSG:4326"
    return transform_geom(src_crs, crs, shapely.geometry.mapping(geom))


def _read(src, bounds, shape, aoi=None):
    """RGB + validity (dataset mask AND AOI) for `bounds` resampled to `shape`.
    Reads only the in-dataset part (boundless reads bypass the overviews)."""
    x0, y0, x1, y1 = bounds
    h, w = shape
    img = np.zeros((3, h, w), np.uint8)
    m = np.zeros((h, w), bool)
    L, B, R, T = src.bounds
    ix0, iy0, ix1, iy1 = max(x0, L), max(y0, B), min(x1, R), min(y1, T)
    if ix1 > ix0 and iy1 > iy0:
        c0 = int(round((ix0 - x0) / (x1 - x0) * w))
        c1 = int(round((ix1 - x0) / (x1 - x0) * w))
        r0 = int(round((y1 - iy1) / (y1 - y0) * h))
        r1 = int(round((y1 - iy0) / (y1 - y0) * h))
        if c1 > c0 and r1 > r0:
            win = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            img[:, r0:r1, c0:c1] = src.read([1, 2, 3], window=win,
                                            out_shape=(3, r1 - r0, c1 - c0),
                                            resampling=Resampling.average)
            m[r0:r1, c0:c1] = src.dataset_mask(
                window=win, out_shape=(r1 - r0, c1 - c0),
                resampling=Resampling.nearest) > 0
    if aoi is not None:
        tr = rasterio.transform.from_bounds(x0, y0, x1, y1, w, h)
        m &= rasterize([aoi], out_shape=(h, w), transform=tr, fill=0,
                       default_value=1, dtype=np.uint8).astype(bool)
    img *= m[None]
    return img, m


def _valid_centres(mask, win_px, min_frac, rng, n):
    """Pick n centres (row, col in mask pixels) for windows of win_px."""
    h, w = mask.shape
    half = max(1, int(round(win_px / 2)))
    frac = uniform_filter(mask.astype(np.float32), size=max(1, int(win_px)),
                          mode="constant")
    good = frac >= min(min_frac, 0.999)
    good[:half, :] = False
    good[-half:, :] = False
    good[:, :half] = False
    good[:, -half:] = False
    rr, cc = np.nonzero(good)
    if len(rr) == 0:
        return None
    idx = rng.choice(len(rr), size=n, replace=len(rr) < n)
    return np.stack([rr[idx], cc[idx]], 1)


def region_of(ortho, aoi):
    src = ortho.src
    if aoi is None:
        return tuple(src.bounds)
    ax0, ay0, ax1, ay1 = shapely.geometry.shape(aoi).bounds
    L, B, R, T = src.bounds
    region = (max(ax0, L), max(ay0, B), min(ax1, R), min(ay1, T))
    if region[2] <= region[0] or region[3] <= region[1]:
        raise ValueError("the area of interest does not overlap the orthophoto")
    return region


def sample_patches(ortho, aoi=None, seed=0, n=N_PATCHES):
    """n random PATCH_PX x PATCH_PX patches at 10 cm inside the AOI and the data
    mask (fully valid where possible, else >= 70 %, >= 30 %, >= 5 %).
    -> (n, 224, 224, 3) uint8, (n,) valid fraction."""
    rng = np.random.default_rng(seed)
    src, scale = ortho.src, ortho.scale
    rx0, ry0, rx1, ry1 = region = region_of(ortho, aoi)
    rw, rh = rx1 - rx0, ry1 - ry0
    m_px = max(rw, rh) / MASK_MAX_PX
    mh, mw = max(1, int(round(rh / m_px))), max(1, int(round(rw / m_px)))
    _, msk = _read(src, region, (mh, mw), aoi)
    if not msk.any():
        raise ValueError("no valid orthophoto pixels inside the area of interest")
    m_px_x, m_px_y = rw / mw, rh / mh
    size = PATCH_PX * PATCH_GSD * scale
    for min_frac in (1.0, 0.7, 0.3, 0.05):
        cen = _valid_centres(msk, size / m_px_x, min_frac, rng, n)
        if cen is not None:
            break
    if cen is None:  # area smaller than one patch: centre on valid pixels
        rr, cc = np.nonzero(msk)
        k = rng.choice(len(rr), size=n, replace=len(rr) < n)
        cen = np.stack([rr[k], cc[k]], 1)
    imgs, fr = [], []
    for rr, cc in cen:
        cx = rx0 + (cc + 0.5) * m_px_x
        cy = ry1 - (rr + 0.5) * m_px_y
        img, m = _read(src, (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
                       (PATCH_PX, PATCH_PX), aoi)
        imgs.append(img.transpose(1, 2, 0))
        fr.append(m.mean())
    return np.stack(imgs), np.array(fr, np.float32)


def on_grid(ortho, aoi, grid):
    """Mean RGB of the orthophoto (inside the AOI) on the Sentinel-2 10 m grid.
    grid: dict(crs, transform, shape) of the cube. -> g10 (3, H, W) float32
    (NaN where empty), g10_valid (H, W) share of valid orthophoto pixels."""
    src, scale = ortho.src, ortho.scale
    rx0, ry0, rx1, ry1 = region_of(ortho, aoi)
    H, W = grid["shape"]
    tr = grid["transform"]
    xs0, ys1 = tr.c, tr.f
    cb = transform_bounds(grid["crs"], src.crs, xs0, ys1 - H * 10, xs0 + W * 10, ys1)
    gb = (max(cb[0], rx0), max(cb[1], ry0), min(cb[2], rx1), min(cb[3], ry1))
    g10 = np.full((3, H, W), np.nan, np.float32)
    gv = np.zeros((H, W), np.float32)
    if gb[2] <= gb[0] or gb[3] <= gb[1]:
        return g10, gv
    # ~1 m ground, at most 3000 px a side
    step = max(1.0 * scale, (gb[2] - gb[0]) / 3000, (gb[3] - gb[1]) / 3000)
    gw = max(1, int(round((gb[2] - gb[0]) / step)))
    gh = max(1, int(round((gb[3] - gb[1]) / step)))
    img, m = _read(src, gb, (gh, gw), aoi)
    src_tr = rasterio.transform.from_bounds(*gb, gw, gh)
    m = m.astype(np.float32)
    stack = np.concatenate([img.astype(np.float32) * m[None], m[None]], 0)
    dst = np.zeros((4, H, W), np.float32)
    reproject(stack, dst, src_transform=src_tr, src_crs=src.crs, dst_transform=tr,
              dst_crs=grid["crs"], resampling=Resampling.average)
    gv = dst[3]
    g10 = np.where(gv > 0, dst[:3] / np.maximum(gv, 1e-6), np.nan).astype(np.float32)
    return g10, gv.astype(np.float32)


def footprint(ortho, aoi):
    """Bounds of the AOI (or the orthophoto) in the orthophoto CRS."""
    return region_of(ortho, aoi)
