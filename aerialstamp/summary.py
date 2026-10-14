"""Calibration and summaries of a day-of-year distribution."""
import numpy as np

from .circular import (DAYS, arc_bins, bin_to_date, circ_diff, circular_median, date_to_bin,
                       hdi_mask, mask_to_arcs, wrapped_gaussian)
from .constants import (DAY_PER_BIN, FLAG_MIN_DAYS, FLAG_PI, MODE_INTERVAL, MODE_MIN_MASS, NBINS,
                        RELIABLE_MAX_WIDTH80)


def calibrate(p, T, s):
    """p_cal ∝ smooth_s(p^(1/T)): temperature T and circular Gaussian smoothing
    of width s bins (the paper's Eq. 4)."""
    q = np.clip(np.asarray(p, np.float64), 1e-12, None)**(1.0 / T)
    q /= q.sum()
    if s > 0:
        k = wrapped_gaussian([-0.5], s)[0]
        q = np.real(np.fft.ifft(np.fft.fft(q) * np.fft.fft(k)))
        q = np.clip(q, 1e-12, None)
        q /= q.sum()
    return q


def n_modes(p):
    """Number of periods of the 90 % interval holding >= 10 % probability."""
    return int(sum(p[arc_bins(a, b)].sum() >= MODE_MIN_MASS
                   for a, b in mask_to_arcs(hdi_mask(p, MODE_INTERVAL)[0])))


def intervals(p, year, masses=(0.5, 0.8, 0.95)):
    out = {}
    for q in masses:
        m = hdi_mask(p, q)[0]
        periods = []
        for a, b in mask_to_arcs(m):
            periods.append(dict(start=str(bin_to_date(a, year)), end=str(bin_to_date(b, year)),
                                days=int(round(len(arc_bins(a, b)) * DAY_PER_BIN)),
                                probability=round(float(p[arc_bins(a, b)].sum()), 4)))
        periods.sort(key=lambda x: -x["probability"])
        out[f"{int(round(q * 100))}%"] = dict(width_days=round(float(m.sum() * DAY_PER_BIN), 1),
                                              periods=periods)
    return out


def plausibility(p, date):
    """pi: total probability of all days at most as likely as the given day (a
    p-value-like score; pi < alpha <=> the date lies outside the 1-alpha interval)."""
    b = int(np.floor(date_to_bin(date))) % NBINS
    return float(p[p <= p[b]].sum())


def summarise(p, year, date=None):
    med = int(circular_median(p)[0])
    w80 = float(hdi_mask(p, 0.8)[0].sum() * DAY_PER_BIN)
    modes = n_modes(p)
    month = np.array([bin_to_date(b, year).month for b in DAYS])
    out = dict(
        predicted_date=str(bin_to_date(med, year)),
        intervals=intervals(p, year),
        n_modes=modes,
        month_probabilities={m: round(float(p[month == m].sum()), 4) for m in range(1, 13)},
        reliable_proposal=bool(modes <= 1 and w80 <= RELIABLE_MAX_WIDTH80),
    )
    if date is not None:
        tb = date_to_bin(date)
        dist = float(abs(circ_diff(med + 0.5, tb)) * DAY_PER_BIN)
        pi = plausibility(p, date)
        out["check"] = dict(
            submitted_date=str(date), plausibility=round(pi, 5),
            days_from_predicted=round(dist, 1),
            flagged=bool(pi < FLAG_PI and dist >= FLAG_MIN_DAYS and modes <= 1),
            rule=(f"flagged if plausibility < {FLAG_PI}, >= {FLAG_MIN_DAYS} days from the "
                  f"predicted date, and only one period of the {int(MODE_INTERVAL * 100)} % "
                  f"interval holds >= {int(MODE_MIN_MASS * 100)} % probability"))
    return out
