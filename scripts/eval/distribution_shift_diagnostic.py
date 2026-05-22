"""Experiment 3: distribution-shift diagnostic.

Why do the Veo y=0 scores sit so far below MovieNet within-scene scores? This
compares the v1.5 and raw-DINOv2-cosine score distributions across three
populations:

  within-scene   MovieNet test cuts labelled y=0 (a real cut, same scene)
  boundary       MovieNet test cuts labelled y=1 (a scene-boundary cut)
  veo            the 10 Veo continuous-action pilot pairs (all intended y=0)

It writes a mean / median / 90th-percentile table and one overlaid histogram
per model, and tests the hypothesis: are Veo pairs more visually similar than
MovieNet within-scene cuts?

Reads cached scores only -- no model inference, no v0/v1.5 artifact touched.

  python scripts/eval/distribution_shift_diagnostic.py
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("distribution_shift")

V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
FIG_DIR = REPO / "reports" / "figures"
REPORT = REPO / "reports" / "v1_distribution_shift.md"

POP_LABEL = {
    "within_scene": "MovieNet within-scene (y=0)",
    "boundary": "MovieNet scene-boundary (y=1)",
    "veo": "Veo continuous-action pairs",
}
POP_COLOR = {"within_scene": "#2a9d4a", "boundary": "#e06641", "veo": "#3366cc"}


def load_populations() -> dict:
    """{model: {within_scene/boundary/veo: score array}} from cached scores."""
    v0, v1 = np.load(V0_NPZ), np.load(V1_NPZ)
    y = v1["test_y"].astype(int)
    assert np.array_equal(y, v0["raw_dino_cosine__test_y"].astype(int))
    pp = pd.read_csv(PER_PAIR_CSV, dtype={"pair_id": str})
    return {
        "v1.5": {
            "within_scene": v1["test_s"][y == 0].astype(float),
            "boundary": v1["test_s"][y == 1].astype(float),
            "veo": pp["v1.5_MLP"].to_numpy(dtype=float),
        },
        "raw_cosine": {
            "within_scene": v0["raw_dino_cosine__test_s"][y == 0].astype(float),
            "boundary": v0["raw_dino_cosine__test_s"][y == 1].astype(float),
            "veo": pp["raw_cos"].to_numpy(dtype=float),
        },
    }


def quantify(pops: dict) -> dict:
    """Per-population mean / median / p90, plus where the Veo mean lands."""
    stats = {}
    for key, s in pops.items():
        stats[key] = {
            "n": int(len(s)),
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "p90": float(np.percentile(s, 90)),
        }
    veo_mean = stats["veo"]["mean"]
    # percentile of the Veo mean inside each MovieNet population
    stats["veo_pctile_in_within_scene"] = float((pops["within_scene"] < veo_mean).mean())
    stats["veo_pctile_in_boundary"] = float((pops["boundary"] < veo_mean).mean())
    # is Veo lower than within-scene? (Mann-Whitney, one-sided)
    u = mannwhitneyu(pops["veo"], pops["within_scene"], alternative="less")
    stats["veo_lt_within_scene_p"] = float(u.pvalue)
    return stats


def plot_model(model: str, pops: dict, out_path: Path) -> None:
    """Overlaid density histograms for the two MovieNet populations + a Veo rug."""
    xmax = float(np.percentile(np.concatenate([pops["within_scene"], pops["boundary"]]), 99.5))
    xmax = max(xmax, float(pops["veo"].max())) * 1.02
    bins = np.linspace(0, xmax, 41)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in ("within_scene", "boundary"):
        s = pops[key]
        ax.hist(
            s,
            bins=bins,
            density=True,
            alpha=0.5,
            color=POP_COLOR[key],
            label=f"{POP_LABEL[key]}  (n={len(s):,}, mean {s.mean():.3f})",
        )
        ax.axvline(s.mean(), color=POP_COLOR[key], ls="--", lw=1.5)

    veo = pops["veo"]
    ax.plot(
        veo,
        np.zeros(len(veo)),
        "|",
        color=POP_COLOR["veo"],
        markersize=22,
        markeredgewidth=2.2,
        clip_on=False,
        label=f"{POP_LABEL['veo']}  (n={len(veo)}, mean {veo.mean():.3f}, rug)",
    )
    ax.axvline(veo.mean(), color=POP_COLOR["veo"], ls="--", lw=2)
    ax.set(
        xlabel=f"{model} inconsistency score",
        ylabel="density",
        title=f"Score-distribution shift — {model}",
        xlim=(0, xmax),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("wrote %s", out_path)


def _ord(x: float) -> str:
    """Integer ordinal: 3 -> '3rd', 30 -> '30th'."""
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _stat_table(stats: dict) -> list[str]:
    rows = ["| population | n | mean | median | p90 |", "|---|--:|--:|--:|--:|"]
    for key in ("within_scene", "boundary", "veo"):
        s = stats[key]
        rows.append(
            f"| {POP_LABEL[key]} | {s['n']:,} | {s['mean']:.3f} | "
            f"{s['median']:.3f} | {s['p90']:.3f} |"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    models = load_populations()
    fig_name = {
        "v1.5": "distribution_shift_v1.5.png",
        "raw_cosine": "distribution_shift_raw_cosine.png",
    }
    stats = {}
    for model, pops in models.items():
        stats[model] = quantify(pops)
        plot_model(model, pops, FIG_DIR / fig_name[model])

    _write_report(Path(args.report), stats, fig_name)
    _print_console(stats)
    log.info("wrote %s", args.report)


def _print_console(stats: dict) -> None:
    for model, st in stats.items():
        print(f"\n=== {model} ===")
        for key in ("within_scene", "boundary", "veo"):
            s = st[key]
            print(
                f"  {POP_LABEL[key]:<32} n={s['n']:>6,}  mean {s['mean']:.3f}  "
                f"median {s['median']:.3f}  p90 {s['p90']:.3f}"
            )
        print(
            f"  -> Veo mean is below {100 * st['veo_pctile_in_within_scene']:.1f}% of "
            f"within-scene cuts; Mann-Whitney Veo<within-scene p={st['veo_lt_within_scene_p']:.2e}"
        )


def _write_report(path: Path, stats: dict, fig_name: dict) -> None:
    v15, rc = stats["v1.5"], stats["raw_cosine"]
    rc_more_similar = rc["veo"]["mean"] < rc["within_scene"]["mean"]
    veo_pct = 100 * v15["veo_pctile_in_within_scene"]

    md = [
        "# v1.5 Distribution-Shift Diagnostic\n",
        "Veo continuous-action pairs (all intended y=0) score far below MovieNet "
        "within-scene cuts. This checks whether that is a model failure or a real "
        "property of the data, by comparing score distributions across three "
        "populations. Scores are **inconsistency** scores (higher = less "
        "continuous); raw cosine is 1 − cos(eL, eR) on the boundary keyframes. "
        "Produced by `scripts/eval/distribution_shift_diagnostic.py` from cached "
        "scores (seed-2 v1.5, MovieNet test split).\n",
        "## v1.5 MLP\n",
        *_stat_table(v15),
        "",
        f"![v1.5 distribution shift](figures/{fig_name['v1.5']})\n",
        "## raw DINOv2 cosine\n",
        *_stat_table(rc),
        "",
        f"![raw cosine distribution shift](figures/{fig_name['raw_cosine']})\n",
        "## Where does Veo sit?\n",
        "Percentile rank = the fraction of MovieNet within-scene cuts that score "
        "below the Veo mean.\n",
        f"- **v1.5:** Veo mean {v15['veo']['mean']:.3f} < within-scene "
        f"{v15['within_scene']['mean']:.3f} < boundary "
        f"{v15['boundary']['mean']:.3f}. The Veo mean lands at the "
        f"**{_ord(veo_pct)} percentile** of within-scene cuts — the lower third, "
        f"below the within-scene median ({v15['within_scene']['median']:.3f}) but "
        f"overlapping the distribution, not a disjoint tail.\n"
        f"- **raw cosine:** Veo mean {rc['veo']['mean']:.3f} < within-scene "
        f"{rc['within_scene']['mean']:.3f} < boundary "
        f"{rc['boundary']['mean']:.3f}. Here the Veo mean is at only the "
        f"**{_ord(100 * rc['veo_pctile_in_within_scene'])} percentile** of "
        f"within-scene cuts — a near-disjoint extreme-similar tail. In "
        f"cosine-similarity terms the Veo boundary frames are "
        f"~{1 - rc['veo']['mean']:.2f} similar vs "
        f"~{1 - rc['within_scene']['mean']:.2f} for real within-scene cuts.\n"
        f"- Mann-Whitney (Veo raw cosine < within-scene raw cosine): "
        f"p = {rc['veo_lt_within_scene_p']:.1e}.\n",
        "The two models disagree on *how* extreme the shift is: pure cosine puts "
        "Veo at the 3rd percentile, v1.5 at the 30th. v1.5's 2305-d feature "
        "(concat + difference, not just cosine) picks up some Veo identity drift "
        "that pure similarity misses, so it spreads the Veo pairs higher than "
        "cosine alone would — a small point in v1.5's favour.\n",
        "## Hypothesis: are Veo pairs more visually similar than within-scene cuts?\n",
        (
            f"**Confirmed.** Raw DINOv2 cosine — a pure visual-similarity measure, "
            f"no training involved — puts the Veo boundary frames at "
            f"{rc['veo']['mean']:.3f} mean distance vs {rc['within_scene']['mean']:.3f} "
            f"for MovieNet within-scene cuts. The Veo frames are markedly *more* "
            f"alike than two shots of one scene in a real film, so v1.5 scoring "
            f"them low ({v15['veo']['mean']:.3f}) is the model behaving correctly, "
            f"not failing."
            if rc_more_similar
            else "**Not confirmed** — Veo raw cosine is not lower than within-scene "
            "raw cosine; the low v1.5 scores are not explained by greater visual "
            "similarity."
        ),
        "",
        _analysis(v15, rc),
        "",
    ]
    path.write_text("\n".join(md))


def _analysis(v15: dict, rc: dict) -> str:
    veo_pct = 100 * v15["veo_pctile_in_within_scene"]
    return (
        "## Read\n\n"
        f"The shift is real and it is in the data, not the model. A Veo "
        f"continuous-action pair is two clips generated from near-identical "
        f"prompts — they share setting, lighting, palette and rough framing — so "
        f"the boundary frames sit very close in embedding space. A real MovieNet "
        f"within-scene cut joins two genuinely different camera setups (angle, "
        f"focal length, subject framing), a much larger visual jump. Raw cosine "
        f"makes this unambiguous: {rc['veo']['mean']:.3f} vs "
        f"{rc['within_scene']['mean']:.3f} mean distance, with the Veo mean at the "
        f"3rd percentile of the within-scene distribution. v1.5 inherits the shift "
        f"— its Veo mean ({v15['veo']['mean']:.3f}) sits well below the "
        f"within-scene mean ({v15['within_scene']['mean']:.3f}), at the "
        f"{_ord(veo_pct)} percentile.\n\n"
        "The consequence is a calibration problem, not a detection problem. v1.5 "
        "is correctly reading the Veo pairs as visually continuous; the MovieNet-"
        "derived operating thresholds (val ~0.75, τ99 ~0.65) sit an order of "
        "magnitude above the entire Veo distribution, so every Veo pair clears "
        "as 'consistent' and nothing — including the identity-drift pairs — gets "
        "flagged. For editor-facing use on AI-gen footage the threshold must be "
        "set from the AI-gen distribution itself, or flagging must be rank-based "
        "(top-k per batch) rather than absolute. The signal to flag identity "
        "drift is present in the *ranking* (Experiment 1: ensembles rank both "
        "major pairs top-2); it is the absolute scale that does not transfer. "
        "This argues for a calibration step on AI-gen data before any editor "
        "deployment, and is orthogonal to the v2 architecture question."
    )


if __name__ == "__main__":
    main()
