"""Plot of the predicted distribution (style of the paper's Fig. 4)."""
import datetime as dt

import numpy as np

from .circular import date_to_bin, hdi_mask
from .constants import NBINS

BLUE, RED, INK, MUTED = "#2E7AC0", "#DC2626", "#1F2937", "#495669"


def plot_result(result, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    year = int(result["provenance"]["year"])
    p = np.array(list(result["daily_probabilities"].values()))
    xs = np.arange(NBINS) + 0.5
    m80 = hdi_mask(p, 0.8)[0]
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    ax.fill_between(xs, 0, np.where(m80, p, 0), color=BLUE, alpha=0.22, lw=0, step="mid")
    ax.plot(xs, p, color=BLUE, lw=1.3)
    top = p.max() * 1.08
    pred = date_to_bin(result["predicted_date"])
    ax.vlines(pred, 0, top, color=INK, lw=1.2, ls="--")
    ax.text(0.01, 0.97, f"predicted {dt.date.fromisoformat(result['predicted_date']):%-d %b %Y}",
            transform=ax.transAxes, color=INK, fontsize=7, va="top")
    if "check" in result:
        sub = date_to_bin(result["check"]["submitted_date"])
        ax.vlines(sub, 0, top, color=RED, lw=1.5)
        c = result["check"]
        ax.text(0.01, 0.84, f"submitted {dt.date.fromisoformat(c['submitted_date']):%-d %b %Y}"
                f"  (plausibility {c['plausibility']:.3g}{', flagged' if c['flagged'] else ''})",
                transform=ax.transAxes, color=RED, fontsize=7, va="top")
    ax.text(0.99, 0.97, "80% interval", transform=ax.transAxes, color=BLUE, fontsize=7,
            ha="right", va="top")
    starts = [date_to_bin(dt.date(year, m, 1)) for m in range(1, 13)]
    ax.set_xticks([s + 15 for s in starts])
    ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=7, color=MUTED)
    ax.set_xticks(starts, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.set_xlim(0, NBINS)
    ax.set_ylim(0, p.max() * 1.62)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title(title or f"{result['model']}, flight year {year}", fontsize=8, color=INK,
                 loc="left")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path
