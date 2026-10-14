import numpy as np

from aerialstamp.features import static_features, week_arrays
from aerialstamp.model import load_ensemble, predict


def test_ensembles_load_and_predict():
    rng = np.random.default_rng(0)
    static = static_features(51.3, 12.3, 2025, 0.03)
    assert static[7] == 0 and static[8] == 0
    emb = rng.normal(size=(16, 1536)).astype(np.float32)
    valid = np.ones(16, bool)
    for kind in ("s2", "image"):
        models, cal = load_ensemble(kind)
        assert len(models) == 5 and {"T", "s"} <= set(cal)
        if kind == "s2":
            w = {"f": rng.normal(size=(50, 32)).astype(np.float32),
                 "pos": np.linspace(-30, 390, 50).astype(np.float32)}
            p = predict(models, static, emb, valid, *week_arrays(w))
        else:
            p = predict(models, static, emb, valid)
        assert p.shape == (365,) and abs(p.sum() - 1) < 1e-6 and (p >= 0).all()
