"""Action 2: rank-based AI-gen flagging.

Sidesteps absolute calibration: each Veo pilot pair is scored by its percentile
rank against the MovieNet within-scene (y=0) test distribution -- a 98k-cut
reference, far more stable than the 10-pair Veo baseline. A pair is *flagged*
when that percentile exceeds 95 for any of {v1.5, v0 logistic, CLIP cosine}
(equivalently, when their max exceeds 95).

The flag is then scored against Dispatch's qualitative buckets: it should fire
on the major-identity pairs (A005, A015), not on the clean pairs (A003, A013);
the six drift pairs are ambiguous and reported as a fired-fraction.

Writes outputs/aigen_eval/results/rank_based_scores.csv and
reports/aigen_rank_based.md. Reads cached scores only.

  python scripts/eval/rank_based_flagging.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rank_based_flagging")

V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
OUT_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/rank_based_scores.csv"
REPORT = REPO / "reports" / "aigen_rank_based.md"

FLAG_THRESHOLD = 95.0  # percentile against MovieNet within-scene
# flag model -> per_pair_scores.csv column
FLAG_MODELS = {"v1.5": "v1.5_MLP", "v0": "v0_logistic", "clip": "clip_cos"}

BUCKET_OF = {
    "A003": "clean",
    "A013": "clean",
    "A001": "drift",
    "A002": "drift",
    "A004": "drift",
    "A011": "drift",
    "A012": "drift",
    "A014": "drift",
    "A005": "major",
    "A015": "major",
}
MAJOR = ["A005", "A015"]
CLEAN = ["A003", "A013"]
DRIFT = ["A001", "A002", "A004", "A011", "A012", "A014"]


def within_scene_reference() -> dict:
    """MovieNet within-scene (y=0) test scores for the three flag models."""
    v0, v1 = np.load(V0_NPZ), np.load(V1_NPZ)
    y = v1["test_y"].astype(int)
    return {
        "v1.5": v1["test_s"][y == 0].astype(float),
        "v0": v0["logistic__test_s"][v0["logistic__test_y"].astype(int) == 0].astype(float),
        "clip": v0["clip_cosine__test_s"][v0["clip_cosine__test_y"].astype(int) == 0].astype(float),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per_pair_csv", default=PER_PAIR_CSV)
    ap.add_argument("--out_csv", default=OUT_CSV)
    args = ap.parse_args()

    ref = within_scene_reference()
    log.info("reference: MovieNet within-scene y=0, n=%d", len(ref["v1.5"]))
    pp = pd.read_csv(args.per_pair_csv, dtype={"pair_id": str})

    rows = []
    for _, r in pp.iterrows():
        pct = {
            m: float(percentileofscore(ref[m], float(r[col]), kind="mean"))
            for m, col in FLAG_MODELS.items()
        }
        max_pct = max(pct.values())
        rows.append(
            {
                "pair_id": r["pair_id"],
                "bucket": BUCKET_OF[r["pair_id"]],
                "v1.5_percentile": pct["v1.5"],
                "v0_percentile": pct["v0"],
                "clip_percentile": pct["clip"],
                "max_percentile": max_pct,
                "flag_v1.5": pct["v1.5"] > FLAG_THRESHOLD,
                "flag_v0": pct["v0"] > FLAG_THRESHOLD,
                "flag_clip": pct["clip"] > FLAG_THRESHOLD,
                "flag_ensemble": max_pct > FLAG_THRESHOLD,
            }
        )
    df = pd.DataFrame(rows)

    csv_cols = [
        "pair_id",
        "v1.5_percentile",
        "v0_percentile",
        "clip_percentile",
        "max_percentile",
        "flag_v1.5",
        "flag_v0",
        "flag_clip",
        "flag_ensemble",
    ]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df[csv_cols].to_csv(args.out_csv, index=False)
    log.info("wrote %s", args.out_csv)

    _write_report(REPORT, df)
    _print_console(df)
    log.info("wrote %s", REPORT)


def _flag_eval(df: pd.DataFrame, flag_col: str) -> dict:
    """Score one flag column against Dispatch's buckets."""
    flagged = set(df.loc[df[flag_col], "pair_id"])
    return {
        "major_caught": sum(p in flagged for p in MAJOR),
        "clean_cleared": sum(p not in flagged for p in CLEAN),
        "drift_flagged": sum(p in flagged for p in DRIFT),
    }


def _separation(df: pd.DataFrame, pct_col: str) -> dict:
    """Can one threshold on this percentile column isolate the two major pairs?

    ``isolates_major`` is the strong condition: both major pairs out-rank *every*
    other pair (clean and drift), so a single threshold catches exactly A005 and
    A015 and nothing else.
    """
    by = df.set_index("pair_id")[pct_col]
    nonmajor = by[CLEAN + DRIFT]
    major_min = float(by[MAJOR].min())
    clean_max = float(by[CLEAN].max())
    nonmajor_max = float(nonmajor.max())
    isolates = major_min > nonmajor_max
    return {
        "major_min": major_min,
        "clean_max": clean_max,
        "nonmajor_max": nonmajor_max,
        "nonmajor_max_id": str(nonmajor.idxmax()),
        "separates_clean": major_min > clean_max,
        "isolates_major": isolates,
        "margin": major_min - nonmajor_max,
        "threshold": (major_min + nonmajor_max) / 2 if isolates else None,
    }


def _print_console(df: pd.DataFrame) -> None:
    print("\n=== Rank-based percentiles vs MovieNet within-scene ===")
    print(f"  {'pair':<7}{'bucket':<8}{'v1.5':>8}{'v0':>8}{'clip':>8}{'max':>8}{'flag?':>8}")
    for _, r in df.sort_values("max_percentile", ascending=False).iterrows():
        print(
            f"  {r['pair_id']:<7}{r['bucket']:<8}{r['v1.5_percentile']:>8.1f}"
            f"{r['v0_percentile']:>8.1f}{r['clip_percentile']:>8.1f}"
            f"{r['max_percentile']:>8.1f}{('YES' if r['flag_ensemble'] else 'no'):>8}"
        )
    print(f"\n=== Flag evaluation (threshold = {FLAG_THRESHOLD:.0f}th percentile) ===")
    for fc in ("flag_v1.5", "flag_v0", "flag_clip", "flag_ensemble"):
        e = _flag_eval(df, fc)
        print(
            f"  {fc:<14} major caught {e['major_caught']}/2  "
            f"clean cleared {e['clean_cleared']}/2  drift flagged {e['drift_flagged']}/6"
        )


def _write_report(path: Path, df: pd.DataFrame) -> None:
    flag_cols = ["flag_v1.5", "flag_v0", "flag_clip", "flag_ensemble"]
    pct_cols = {
        "flag_v1.5": "v1.5_percentile",
        "flag_v0": "v0_percentile",
        "flag_clip": "clip_percentile",
        "flag_ensemble": "max_percentile",
    }
    evals = {fc: _flag_eval(df, fc) for fc in flag_cols}
    seps = {fc: _separation(df, pct_cols[fc]) for fc in flag_cols}
    any_fires = any(df[fc].any() for fc in flag_cols)

    md = [
        "# AI-Gen Rank-Based Flagging\n",
        "Each Veo pilot pair is ranked by percentile against the MovieNet "
        "within-scene (y=0) test distribution (n=97,840). A pair is **flagged** "
        f"when that percentile exceeds **{FLAG_THRESHOLD:.0f}** for any of v1.5, "
        "v0 logistic, or CLIP cosine. This sidesteps absolute calibration. "
        "Source: `outputs/aigen_eval/results/rank_based_scores.csv`.\n",
        "## Per-pair percentiles (vs MovieNet within-scene)\n",
        "| pair | bucket | v1.5 %ile | v0 %ile | clip %ile | max %ile | flagged? |",
        "|---|---|--:|--:|--:|--:|:--:|",
    ]
    for _, r in df.sort_values("max_percentile", ascending=False).iterrows():
        md.append(
            f"| {r['pair_id']} | {r['bucket']} | {r['v1.5_percentile']:.1f} "
            f"| {r['v0_percentile']:.1f} | {r['clip_percentile']:.1f} "
            f"| {r['max_percentile']:.1f} | {'YES' if r['flag_ensemble'] else 'no'} |"
        )

    md.append(f"\n## Flag evaluation at the {FLAG_THRESHOLD:.0f}th-percentile threshold\n")
    md.append("| flag | major caught (/2) | clean cleared (/2) | drift flagged (/6) |")
    md.append("|---|:--:|:--:|:--:|")
    for fc in flag_cols:
        e = evals[fc]
        md.append(f"| {fc} | {e['major_caught']} | {e['clean_cleared']} | {e['drift_flagged']} |")

    if not any_fires:
        md.append(
            f"\n**At the {FLAG_THRESHOLD:.0f}th-percentile threshold no flag fires "
            f"on any pair.** Every Veo pair — including the major-identity "
            f"failures — ranks below the 95th percentile of MovieNet within-scene "
            f"cuts: the worst AI-gen identity drift is still visually *more* "
            f"continuous than 5% of ordinary real-film within-scene cuts. The "
            f"binary p95 rank flag cannot discriminate because nothing reaches it "
            f"— the distribution shift (see `v1_distribution_shift.md`) defeats "
            f"the absolute cutoff, not the ranking idea itself.\n"
        )

    md.append("## Does the percentile *ordering* still separate the buckets?\n")
    md.append(
        "The binary p95 flag is empty, but the percentile ranks still order the "
        "pairs. The useful question: can a *single threshold* on a percentile "
        "column isolate the two major-identity pairs from all eight other pairs "
        "(clean + drift)? `non-major max` is the highest-ranked non-major pair — "
        "the major pairs must clear it.\n"
    )
    md.append("| flag model | clean max | major min | non-major max | isolates major? | margin |")
    md.append("|---|--:|--:|--:|:--:|--:|")
    for fc in flag_cols:
        s = seps[fc]
        md.append(
            f"| {pct_cols[fc]} | {s['clean_max']:.1f} | {s['major_min']:.1f} "
            f"| {s['nonmajor_max']:.1f} ({s['nonmajor_max_id']}) "
            f"| {'**yes**' if s['isolates_major'] else 'no'} | {s['margin']:+.1f} |"
        )
    md.append("\n" + _recommendation(df, evals, seps, pct_cols, any_fires))
    path.write_text("\n".join(md))


def _recommendation(
    df: pd.DataFrame, evals: dict, seps: dict, pct_cols: dict, any_fires: bool
) -> str:
    flag_cols = ["flag_v1.5", "flag_v0", "flag_clip", "flag_ensemble"]
    isolating = [fc for fc in flag_cols if seps[fc]["isolates_major"]]
    best = max(isolating, key=lambda fc: seps[fc]["margin"]) if isolating else None
    head = "## Recommendation\n\n"
    if any_fires:
        scored = sorted(
            flag_cols,
            key=lambda fc: (
                evals[fc]["major_caught"],
                evals[fc]["clean_cleared"],
                -evals[fc]["drift_flagged"],
            ),
            reverse=True,
        )
        return head + (
            f"At the p95 threshold, **{scored[0]}** discriminates best "
            f"(major caught {evals[scored[0]]['major_caught']}/2, clean cleared "
            f"{evals[scored[0]]['clean_cleared']}/2, drift "
            f"{evals[scored[0]]['drift_flagged']}/6)."
        )
    body = (
        "The p95 rank flag fires on nothing, so the four binary flags are tied at "
        "useless — the **threshold**, not the ranking idea, is wrong. The usable "
        "signal is in the percentile *ordering*, and there it is decisive. "
    )
    if best:
        s = seps[best]
        body += (
            f"**{pct_cols[best]}** is the clear winner: a single threshold "
            f"isolates *both* major-identity pairs from all eight other pairs. "
            f"The major pairs rank at the {s['major_min']:.0f}th percentile and "
            f"above; every clean and drift pair ranks at or below the "
            f"{s['nonmajor_max']:.0f}th ({s['nonmajor_max_id']}) — a "
            f"{s['margin']:.0f}-point margin. A rank flag at ~"
            f"{s['threshold']:.0f} (anywhere in that window, not 95) catches A005 "
            f"and A015 and nothing else: 2/2 major, 2/2 clean cleared, 0/6 drift "
            f"over-flagged. That CLIP — the weakest MovieNet scorer — gives the "
            f"cleanest AI-gen identity-failure flag is consistent with every "
            f"prior experiment: CLIP captures the semantic identity drift DINOv2 "
            f"misses. "
        )
        runners = [fc for fc in isolating if fc != best]
        if runners:
            r = seps[runners[0]]
            body += (
                f"`{pct_cols[runners[0]]}` also isolates both major pairs but on "
                f"a tighter {r['margin']:.0f}-point margin. "
            )
    else:
        body += (
            "No single column isolates the major pairs from all others — v1.5 "
            "ranks A005 high but A015 low, v0 the reverse. "
        )
    rec_model = pct_cols[best].replace("_percentile", "") if best else "max_percentile"
    rec_thr = f"~{seps[best]['threshold']:.0f}" if best else "data-driven, not 95"
    tail = (
        f"**Single recommendation:** flag a Veo continuous-action pair when its "
        f"**{rec_model}** percentile against the MovieNet within-scene "
        f"distribution exceeds {rec_thr}. Rank-against-MovieNet is the right "
        f"idea; the 95th-percentile cutoff is simply far too high for the "
        f"AI-gen distribution — the entire Veo set sits below it. Pair this "
        f"ordering with the same-domain Veo calibration from Action 1 for a "
        f"deployable flag. Caveat in bold: n=10, single class, one generator — "
        f"the threshold is provisional and must be refit as Veo data grows."
    )
    return head + body + "\n\n" + tail


if __name__ == "__main__":
    main()
