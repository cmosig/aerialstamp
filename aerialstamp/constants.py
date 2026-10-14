"""Fixed conventions of the AerialStamp model (identical to the training code)."""

NBINS = 365  # day bins on a circle; leap years are squeezed by one day
DAY_PER_BIN = 365.25 / NBINS

# orthophoto views
PATCH_PX = 224  # patch side in pixels
PATCH_GSD = 0.10  # ground sampling distance of the patches [m]
N_PATCHES = 16
MASK_MAX_PX = 2048  # resolution of the validity mask used to place patches
BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"  # timm, frozen

# Sentinel-2
S2_RES = 10.0
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
         "B11", "B12"]
MATCH_NAMES = [
    "n_frac",  # share of orthophoto-covered AOI pixels clear this week
    "r_bright",  # corr(ortho mean RGB, S2 B02+B03+B04)
    "r_red", "r_green", "r_blue",  # per-band correlations
    "r_gchroma",  # corr(G/(R+G+B), B03/(B02+B03+B04))
    "r_grvi",  # corr((G-R)/(G+R), (B03-B04)/(B03+B04))
    "r_grvi_ndvi",  # corr(ortho GRVI, S2 NDVI)
    "r_bright_nir",  # corr(ortho brightness, B08)
    "d_gchroma",  # S2 green chromaticity - ortho green chromaticity (AOI means)
    "d_rchroma",
    "r_rank",  # Spearman correlation of brightness
]
MATCH_MAX_SIDE = 160  # AOI pixels per side used for the match features (subsampled above)
MATCH_MIN_PIX = 12  # minimum matched pixels for a correlation
WINDOW_PAD = 35  # days of the neighbouring years in the week window
N_WEEKS = 64  # maximum number of week tokens
SENTLE_ORIGIN = "2015-07-01"  # weekly composite grid as in the training cubes

# static site facts: columns 7 and 8 (Sentinel-2 / match-feature availability)
# were removed from the model (data leakage) and are always zero
STATIC_MASKED = (7, 8)

# flagging rule of the paper (Section 3.8)
FLAG_PI = 0.01
FLAG_MIN_DAYS = 60
MODE_INTERVAL = 0.90
MODE_MIN_MASS = 0.10
RELIABLE_MAX_WIDTH80 = 90
