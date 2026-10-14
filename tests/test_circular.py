import numpy as np

from aerialstamp.circular import (bin_to_date, circular_median, date_to_bin, hdi_mask,
                                 mask_to_arcs, wrapped_gaussian)
from aerialstamp.summary import calibrate, n_modes, plausibility, summarise


def test_dates_roundtrip():
    for d in ("2023-01-01", "2023-06-15", "2024-12-31", "2024-02-29"):
        b = date_to_bin(d)
        assert 0 <= b < 365
        assert abs((bin_to_date(b, int(d[:4])) - np.datetime64(d).astype(object)).days) <= 1


def test_interval_and_plausibility():
    p = wrapped_gaussian([180.0], 10.0)[0]
    assert abs(circular_median(p)[0] - 180) <= 1
    m80 = hdi_mask(p, 0.8)[0]
    assert len(mask_to_arcs(m80)) == 1 and abs(p[m80].sum() - 0.8) < 0.02
    # a date outside the 99 % interval has plausibility < 0.01
    assert plausibility(p, "2023-01-15") < 0.01
    assert plausibility(p, bin_to_date(180, 2023)) > 0.9


def test_modes_and_flag():
    p = 0.5 * wrapped_gaussian([90.0], 8.0)[0] + 0.5 * wrapped_gaussian([270.0], 8.0)[0]
    assert n_modes(p) == 2
    s = summarise(p, 2023, date=bin_to_date(10, 2023))
    assert s["check"]["flagged"] is False  # never flagged with two modes
    q = wrapped_gaussian([200.0], 10.0)[0]
    s = summarise(q, 2023, date=bin_to_date(30, 2023))
    assert s["check"]["flagged"] is True


def test_calibration_flattens():
    p = wrapped_gaussian([100.0], 5.0)[0]
    q = calibrate(p, 2.0, 0)
    assert abs(q.sum() - 1) < 1e-9 and q.max() < p.max()
