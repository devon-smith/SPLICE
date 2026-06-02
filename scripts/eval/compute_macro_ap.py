"""Compute macro per-movie average precision (the literature-aligned MovieNet metric).

MovieNet scene-segmentation papers (ShotCoL, BaSSL, TranS4mer, MEGA, MASRC,
MHRT, NeighborNet) report AP as the *mean of per-movie AP values*, not pooled
AP over every test pair. The numeric gap matters: pooled AP gives one number
weighted by movie size, macro AP weighs every movie equally. This script
recomputes the canonical metric and a movie-level paired bootstrap CI.

  # single model
  python compute_macro_ap.py --scores PATH --cut_index PATH --out reports/macro_ap_<m>.json

  # paired comparison
  python compute_macro_ap.py --compare SCORES_A SCORES_B --cut_index PATH \\
      --out reports/macro_ap_compare.json

Scores arrays from all three trained heads (v0, v1.5, v2) are row-aligned to
the canonical test order (pairs/dino_v0_boundary/meta.parquet[split=='test'],
which preserves cuts.parquet order); pass cuts.parquet as --cut_index. Reads
cached scores only; trains nothing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_macro_ap")


def compute_macro_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    movie_ids: np.ndarray,
    min_cuts: int = 5,
) -> dict:
    """Macro per-movie average precision.

    For each unique movie_id, computes sklearn average_precision_score on that
    movie's slice. Skips movies with <``min_cuts`` cuts, 0 positives, or all
    positives (AP is undefined / degenerate in those cases).

    Returns a dict with ``macro_ap`` (mean over kept movies), ``median_ap``,
    ``std_ap`` (sample std), ``n_movies_used``, ``n_movies_skipped`` and the
    per-movie AP table.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    movie_ids = np.asarray(movie_ids)
    per_movie: dict[str, dict] = {}
    skipped: list[dict] = []
    for mid in np.unique(movie_ids):
        idx = movie_ids == mid
        n = int(idx.sum())
        pos = int(labels[idx].sum())
        if n < min_cuts or pos == 0 or pos == n:
            skipped.append({"movie_id": str(mid), "n": n, "pos": pos})
            continue
        per_movie[str(mid)] = {
            "ap": float(average_precision_score(labels[idx], scores[idx])),
            "n": n,
            "pos": pos,
        }
    aps = np.array([m["ap"] for m in per_movie.values()], dtype=float)
    return {
        "macro_ap": float(aps.mean()) if len(aps) else float("nan"),
        "median_ap": float(np.median(aps)) if len(aps) else float("nan"),
        "std_ap": float(aps.std(ddof=1)) if len(aps) > 1 else float("nan"),
        "n_movies_used": len(per_movie),
        "n_movies_skipped": len(skipped),
        "per_movie_ap": per_movie,
        "skipped_movies": skipped,
    }


def movie_level_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    movie_ids: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    min_cuts: int = 5,
) -> dict:
    """Paired bootstrap CI for macro AP(A) − macro AP(B) by resampling movies.

    Pair-level resampling is invalid here because cuts from the same movie are
    correlated; movie-level resampling respects the clustering. Each bootstrap
    draw resamples 64 movies with replacement and computes macro AP for both
    models on that resample (a movie picked twice contributes twice, equal
    weight per pick).
    """
    # per-movie AP is deterministic, so precompute it once for each model and
    # resample over the AP vectors -- exactly equivalent to resampling movies
    # and re-scoring, but ~1000x faster.
    res_a = compute_macro_ap(scores_a, labels, movie_ids, min_cuts)
    res_b = compute_macro_ap(scores_b, labels, movie_ids, min_cuts)
    common = sorted(set(res_a["per_movie_ap"]) & set(res_b["per_movie_ap"]))
    ap_a = np.array([res_a["per_movie_ap"][m]["ap"] for m in common])
    ap_b = np.array([res_b["per_movie_ap"][m]["ap"] for m in common])
    obs_a, obs_b = float(ap_a.mean()), float(ap_b.mean())
    rng = np.random.default_rng(seed)
    n = len(common)
    deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        deltas[b] = float(ap_a[pick].mean() - ap_b[pick].mean())
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "delta_mean": float(obs_a - obs_b),
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool((lo > 0) or (hi < 0)),
        "macro_ap_a": obs_a,
        "macro_ap_b": obs_b,
        "n_resamples": int(n_resamples),
        "n_movies": n,
    }


def load_scores(
    npz_path: str | Path,
    score_key: str | None = None,
    label_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load (test scores, test labels) from an npz; auto-detect keys when possible."""
    f = np.load(str(npz_path))
    files = set(f.files)
    if score_key is None:
        if "test_s" in files:
            score_key = "test_s"
        else:
            cands = sorted(k for k in files if k.endswith("__test_s"))
            if len(cands) == 1:
                score_key = cands[0]
            else:
                raise SystemExit(
                    f"{npz_path}: cannot auto-detect score key (found {sorted(files)}); "
                    f"pass --score_key"
                )
    if label_key is None:
        if "test_y" in files:
            label_key = "test_y"
        else:
            base = score_key.replace("__test_s", "__test_y")
            if base in files:
                label_key = base
            else:
                raise SystemExit(f"{npz_path}: cannot auto-detect label key for {score_key}")
    return f[score_key].astype(float), f[label_key].astype(int), score_key, label_key


def load_movie_ids(cut_index_path: str, n_test: int) -> np.ndarray:
    """Test-split movie_ids in the canonical training/scoring row order."""
    df = pd.read_parquet(cut_index_path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if len(test) != n_test:
        raise SystemExit(
            f"length mismatch: cut_index test rows {len(test)} vs scores n_test {n_test} -- "
            f"the scores npz may not be aligned to this cut index"
        )
    return test["movie_id"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scores", help="npz path -- compute macro AP for one model")
    g.add_argument(
        "--compare",
        nargs=2,
        metavar=("SCORES_A", "SCORES_B"),
        help="two npz paths -- paired movie-level bootstrap of AP(A) - AP(B)",
    )
    ap.add_argument("--cut_index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score_key", default=None, help="key in --scores npz (auto-detect default)")
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--score_key_a", default=None, help="(compare) key in SCORES_A")
    ap.add_argument("--score_key_b", default=None, help="(compare) key in SCORES_B")
    ap.add_argument("--label_key_a", default=None)
    ap.add_argument("--label_key_b", default=None)
    ap.add_argument("--n_resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_cuts", type=int, default=5)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.scores:
        scores, labels, sk, lk = load_scores(args.scores, args.score_key, args.label_key)
        movie_ids = load_movie_ids(args.cut_index, len(scores))
        res = compute_macro_ap(scores, labels, movie_ids, args.min_cuts)
        res.update(
            {
                "npz_path": args.scores,
                "score_key": sk,
                "label_key": lk,
                "pooled_auprc": float(average_precision_score(labels, scores)),
                "n_test_cuts": int(len(scores)),
                "min_cuts_threshold": args.min_cuts,
            }
        )
        Path(args.out).write_text(json.dumps(res, indent=2))
        log.info(
            "macro_ap=%.4f median=%.4f std=%.4f n_movies=%d (skipped %d) "
            "pooled_auprc=%.4f -> %s",
            res["macro_ap"],
            res["median_ap"],
            res["std_ap"],
            res["n_movies_used"],
            res["n_movies_skipped"],
            res["pooled_auprc"],
            args.out,
        )
    else:
        a_path, b_path = args.compare
        sa, ya, ska, lka = load_scores(a_path, args.score_key_a, args.label_key_a)
        sb, yb, skb, lkb = load_scores(b_path, args.score_key_b, args.label_key_b)
        if not np.array_equal(ya, yb):
            raise SystemExit(
                "label arrays disagree -- the two score files must share the same test split"
            )
        movie_ids = load_movie_ids(args.cut_index, len(sa))
        boot = movie_level_bootstrap(
            sa, sb, ya, movie_ids, args.n_resamples, args.seed, args.min_cuts
        )
        boot.update(
            {
                "a_path": a_path,
                "b_path": b_path,
                "score_key_a": ska,
                "score_key_b": skb,
                "n_test_cuts": int(len(sa)),
            }
        )
        Path(args.out).write_text(json.dumps(boot, indent=2))
        log.info(
            "delta=%+.4f CI [%+.4f, %+.4f] excludes_zero=%s -> %s",
            boot["delta_mean"],
            boot["ci_low"],
            boot["ci_high"],
            boot["excludes_zero"],
            args.out,
        )


if __name__ == "__main__":
    main()
