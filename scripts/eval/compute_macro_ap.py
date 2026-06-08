# macro per-movie AP -- the metric MovieNet papers report

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# mean of per-movie AP; skip movies with <min_cuts or all-one-class
def macro_ap(scores, labels, movie_ids, min_cuts=5):
    per_movie = {}
    for mid in np.unique(movie_ids):
        m = movie_ids == mid
        y = labels[m]
        if m.sum() < min_cuts or y.sum() == 0 or y.sum() == m.sum():
            continue
        per_movie[str(mid)] = float(average_precision_score(y, scores[m]))
    aps = np.array(list(per_movie.values()))
    return {
        "macro_ap": aps.mean(),
        "median_ap": np.median(aps),
        "std_ap": float(aps.std(ddof=1)),
        "n_movies_used": len(aps),
        "per_movie_ap": per_movie,
    }


# paired movie-level bootstrap on macro_ap(A) - macro_ap(B). resample movies,
# not pairs, since cuts inside a movie are correlated. per-movie AP is
# deterministic so we resample the precomputed AP vector instead of rescoring.
def bootstrap_gap(scores_a, scores_b, labels, movie_ids, n_resamples=1000, seed=42):
    a = macro_ap(scores_a, labels, movie_ids)
    b = macro_ap(scores_b, labels, movie_ids)
    movies = sorted(set(a["per_movie_ap"]) & set(b["per_movie_ap"]))
    ap_a = np.array([a["per_movie_ap"][m] for m in movies])
    ap_b = np.array([b["per_movie_ap"][m] for m in movies])

    rng = np.random.default_rng(seed)
    n = len(movies)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = ap_a[idx].mean() - ap_b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    return {
        "delta_mean": float(ap_a.mean() - ap_b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "macro_ap_a": ap_a.mean(),
        "macro_ap_b": ap_b.mean(),
        "n_resamples": n_resamples,
        "n_movies": n,
    }


def load_test_movie_ids(cut_index_path, expected_n):
    df = pd.read_parquet(cut_index_path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    assert len(test) == expected_n, f"cut_index test rows {len(test)} != scores n {expected_n}"
    return test["movie_id"].to_numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scores", help="one npz path -- macro AP for a single model")
    g.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="two npz paths -- paired bootstrap of A vs B")
    ap.add_argument("--cut_index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score_key", default="test_s")
    ap.add_argument("--label_key", default="test_y")
    ap.add_argument("--score_key_a", default=None)
    ap.add_argument("--score_key_b", default=None)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.scores:
        f = np.load(args.scores)
        s = f[args.score_key].astype(float)
        y = f[args.label_key].astype(int)
        movie_ids = load_test_movie_ids(args.cut_index, len(s))
        result = macro_ap(s, y, movie_ids)
        result["pooled_auprc"] = float(average_precision_score(y, s))
        result["npz_path"] = args.scores
        result["score_key"] = args.score_key
        print(f"macro_ap={result['macro_ap']:.4f}  median={result['median_ap']:.4f}  "
              f"std={result['std_ap']:.4f}  n={result['n_movies_used']}  "
              f"pooled={result['pooled_auprc']:.4f}  -> {args.out}")
    else:
        fa = np.load(args.compare[0])
        fb = np.load(args.compare[1])
        ka = args.score_key_a or args.score_key
        kb = args.score_key_b or args.score_key
        sa, sb = fa[ka].astype(float), fb[kb].astype(float)
        ya, yb = fa[args.label_key].astype(int), fb[args.label_key].astype(int)
        assert np.array_equal(ya, yb), "label arrays disagree across the two npz files"
        movie_ids = load_test_movie_ids(args.cut_index, len(sa))
        result = bootstrap_gap(sa, sb, ya, movie_ids)
        result["a_path"], result["b_path"] = args.compare
        result["score_key_a"], result["score_key_b"] = ka, kb
        print(f"delta={result['delta_mean']:+.4f}  "
              f"CI=[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
              f"excludes_zero={result['excludes_zero']}  -> {args.out}")

    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
