"""Command line: `aerialstamp estimate` and `aerialstamp check`."""
import argparse
import json
import sys


def _common(ap):
    ap.add_argument("orthophoto", help="RGB GeoTIFF of the orthophoto")
    ap.add_argument("--aoi", help="vector file with the area of interest (optional)")
    ap.add_argument("--mode", choices=("auto", "s2", "image"), default="auto",
                    help="auto: S2 model if Sentinel-2 weeks with match features exist, "
                         "else the image-only model (default)")
    ap.add_argument("--cube", help="existing sentle cube (zarr) instead of a download")
    ap.add_argument("--no-download", action="store_true",
                    help="never download Sentinel-2 (auto mode falls back to image-only)")
    ap.add_argument("--cache-dir", help="where downloaded cubes are kept")
    ap.add_argument("--s2-start", help="start of the Sentinel-2 download (default: year - 42 d)")
    ap.add_argument("--s2-end", help="end of the Sentinel-2 download (default: year + 42 d)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--workers", type=int, default=4, help="sentle download workers")
    ap.add_argument("--seed", type=int, default=0, help="seed of the patch positions")
    ap.add_argument("--lat", type=float, help="site latitude (default: orthophoto centre)")
    ap.add_argument("--lon", type=float, help="site longitude (default: orthophoto centre)")
    ap.add_argument("--out", help="write the full result as JSON")
    ap.add_argument("--plot", help="write a plot of the distribution (PNG/PDF)")
    ap.add_argument("--quiet", action="store_true", help="print only the JSON summary")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="aerialstamp",
        description="Estimate the acquisition date of a drone orthophoto as a calibrated "
                    "distribution over the days of the flight year.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    est = sub.add_parser("estimate", help="estimate the flight date within a given year")
    _common(est)
    est.add_argument("--year", type=int, required=True, help="flight year")
    chk = sub.add_parser("check", help="check a submitted date (plausibility, flag)")
    _common(chk)
    chk.add_argument("--date", required=True, help="submitted date YYYY-MM-DD")
    a = ap.parse_args(argv)

    from .core import date_orthophoto
    res = date_orthophoto(
        a.orthophoto, year=getattr(a, "year", None), date=getattr(a, "date", None), aoi=a.aoi,
        mode=a.mode, cube=a.cube, download_s2=not a.no_download, cache_dir=a.cache_dir,
        s2_start=a.s2_start, s2_end=a.s2_end, device=a.device, seed=a.seed,
        num_workers=a.workers, lat=a.lat, lon=a.lon)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    if a.plot:
        from .plot import plot_result
        plot_result(res, a.plot)
    short = {k: res[k] for k in ("model", "predicted_date", "n_modes", "reliable_proposal")}
    short["80% interval"] = res["intervals"]["80%"]
    if "check" in res:
        short["check"] = {k: v for k, v in res["check"].items() if k != "rule"}
    short["sentinel2"] = res["provenance"]["sentinel2"]
    if res["provenance"]["notes"]:
        short["notes"] = res["provenance"]["notes"]
    json.dump(short, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
