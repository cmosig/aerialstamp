# Validation against the research pipeline (2026-10-01)

All runs on phaestos (scienceagent environment, Python 3.12, torch 2.10, timm 1.0.24,
sentle 2025.12.1 from the local checkout), GPU 3 unless noted. Orthophotos are the
deadtrees.earth COGs used in the paper; the research features are those of the paper
(`labels.parquet`, `ortho/<id>.npz`, `dataset_dinov2b.npz`, `numbers/nw_*_oof_cal.npy`).
The comparison script is `paper_isprs/analysis/dd_validate.py` (it calls
`aerialstamp.date_orthophoto` with the metadata AOI, seed = dataset id and the label
coordinates, and compares its inputs with the stored research features).

Note: the weights are refits on all labelled orthophotos, so these orthophotos were part of
their training data; the paper's numbers are out-of-fold. The comparison checks that the
package reproduces the research inputs and gives sensible dates, not its accuracy.

## 1. Training cube (`--cube`), S2 model — dataset 6148 (Leipzig, submitted 2025-03-19)

```bash
PYTHONPATH=../..:../aerialstamp CUDA_VISIBLE_DEVICES=3 python dd_validate.py 6148
```

| | result |
|---|---|
| 16 patches | identical to `ortho/6148.npz` (max abs diff 0) |
| DINOv2 embeddings | identical to `p10_emb` (max rel diff 0.0) |
| site facts | identical (columns 7, 8 zero) |
| week tokens | 62 = 62 weeks; max abs diff 3.0e-4 (float16 storage in the dataset) |
| predicted date | 2025-03-21 (80 % interval 2025-03-03 – 2025-04-05) |
| paper, out-of-fold | 2025-03-24 |

## 2. Image-only mode — dataset 6148

```bash
PYTHONPATH=../..:../aerialstamp CUDA_VISIBLE_DEVICES=3 python dd_validate.py 6148 --image
```

Predicted 2025-03-06; 80 % interval 2025-01-07 – 2025-05-24 (75 %) plus 2025-11-13 – 2025-12-14
(5 %); paper out-of-fold image-only prediction 2025-03-28. Patches, embeddings and site facts
identical to the research features.

CPU only, through the CLI, default seed and no AOI (8.3 s wall, 1.5 GB RAM):

```bash
CUDA_VISIBLE_DEVICES="" python -m aerialstamp estimate 6148_….tif --year 2025 --mode image
```

Predicted 2025-03-08; 80 % interval 2025-01-09 – 2025-05-28 plus 2025-11-17 – 2025-12-10.

## 3. Check mode — dataset 1608 (Switzerland, submitted 2021-03-24; Fig. 4b of the paper)

```bash
PYTHONPATH=../..:../aerialstamp CUDA_VISIBLE_DEVICES=3 python dd_validate.py 1608 --date 2021-03-24
```

Predicted 2021-08-16 (80 % interval 2021-07-15 – 2021-09-08; paper out-of-fold 2021-08-21);
plausibility of the submitted date 0.00011, 145.6 days from the predicted date, **flagged**
(as in the paper). Week tokens: max abs diff 5.1e-4. Plot: `docs/example_1608_flagged.png`.

## 4. Fresh Sentinel-2 download with sentle — dataset 6161 (Leipzig, submitted 2025-07-08)

```bash
CUDA_VISIBLE_DEVICES=3 python -m aerialstamp check 6161_auwald_timeseries21__08.07.2025.zip.tif \
    --date 2025-07-08 --aoi aoi_6161.geojson --device cuda --workers 8 \
    --cache-dir cache --out fresh_6161.json --plot fresh_6161.png
```

6 min wall in total (64 weekly composites 2024-11-21 – 2026-02-05 over 98 × 101 cells (7.8 MB), cloud
classifier on the GPU; almost all of it the download).

```text
model: S2 model; predicted 2025-07-06; 80 % interval 2025-06-25 – 2025-07-17 (23 days)
check: plausibility 0.847, 1.5 days from the predicted date, not flagged
Sentinel-2: 62 weeks in the window, 42 with match features, 40 mostly clear, last week 2026-02-05
paper out-of-fold prediction: 2025-07-06
```

Features from the fresh cube against the training cube of the same orthophoto
(`dd_validate.py 6161 --cube cache/…zarr`, same patches): 62 = 62 weeks; one week differs in its
clear/match flag (cloud mask of the newer sentle); median abs difference 0.0002 in the band
tokens and 0.0014 in the match features; the AOI covers 5,146 instead of 4,963 cells (the AOI
is rasterised by the package, the training cube carries its own AOI raster). Predicted date
with the fresh cube 2025-07-07, with the training cube 2025-07-06. Plot:
`docs/example_6161_check.png`.

## 5. Unit tests

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests   # 5 passed
```

## Issue found and fixed

sentle forks worker processes; once the parent had initialised CUDA (DINOv2 on the GPU), the
cloud classifier in the workers failed with "Cannot re-initialize CUDA in forked subprocess" and
the download hung. The package now downloads Sentinel-2 before anything touches the GPU, and
writes the cube to `<name>.zarr.part` first, so only complete downloads are cached.

## Version 1.1.0 (2026-10-08): weights refit on the extended set

The weights were refit on the 8,825 labelled orthophotos that now have a Sentinel-2 series
(version 1.0.0: 5,299), with the calibration of the paper's new cross-validation (T = 2.0 for
both models). The sections above describe version 1.0.0; the fresh-download and other checks
there were not repeated. Repeated for dataset 6148 (Leipzig, submitted 2025-03-19) with the
training cube:

| | S2 model | image-only model |
|---|---|---|
| 16 patches, DINOv2 embeddings, site facts | identical | identical |
| week tokens | 62 = 62 weeks; max abs diff 3.0e-4 | – |
| predicted date | 2025-03-22 | 2025-03-14 |
| paper, out-of-fold | 2025-03-18 | 2025-02-25 |

## Version 1.2.0 (2026-10-08): weights refit on public orthophotos only

The paper now uses only the orthophotos that are public in deadtrees.earth. The weights
were refit on these 6,398 labelled orthophotos; calibration from the paper's
cross-validation (T = 2.0 for the S2 model; T = 1.7 and 10-day smoothing for the image-only
model). Repeated for dataset 6148 (Leipzig, submitted 2025-03-19) with the training cube:
inputs identical to the research pipeline as before; predicted 2025-03-20 (S2 model;
paper out-of-fold 2025-03-24) and 2025-03-12 (image-only model; out-of-fold 2025-03-18).
A benchmark with the data and evaluation protocol is at
https://github.com/cmosig/aerialstamp-benchmark.

## Version 1.3.0 (2026-10-08): public CC BY orthophotos only

The paper and the benchmark (https://github.com/cmosig/aerialstamp-benchmark) use the
6,984 orthophotos that are public under CC BY. The weights were refit on their 6,273
labelled orthophotos; calibration from the paper's cross-validation (T = 2.0 for the S2
model; T = 1.7 and 10-day smoothing for the image-only model, chosen in four of five
folds). Dataset 6148 with the training cube: inputs identical to the research pipeline;
predicted 2025-03-21 (S2 model; paper out-of-fold 2025-03-25) and 2025-03-13 (image-only
model; out-of-fold 2025-03-20).

## Version 2.0.0 (2026-10-09): renamed from DroneDater to AerialStamp

Name only (package, import, command line); weights and behaviour as in version 1.3.0.
