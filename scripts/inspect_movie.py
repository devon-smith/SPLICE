"""Inspect one MovieNet movie's annotations to verify our parsing assumptions.

Diagnostic only -- prints to stdout for a human; produces no project artifacts.

Discovered data layout (BaSSL MovieNet distribution, verified May 2026)
----------------------------------------------------------------------
<data_root>/
  anno/anno.{train,val,test}.ndjson   one JSON object PER SHOT, fields:
      video_id          IMDb id, e.g. "tt0048545"
      shot_id           zero-padded 4-digit shot index, e.g. "0042"
      num_shot          total shots in the movie
      boundary_label    1 iff the cut AFTER this shot is a scene boundary, i.e.
                        this shot is the last shot of its scene (== BaSSL "label318")
      invideo_scene_id  scene index within the movie (0-based)
      global_scene_id / global_video_id   dataset-global ids
    train/val/test = 190/64/64 movies (the 318-movie scene-seg subset);
    a movie's split == whichever ndjson file it appears in.
  240P_frames/<video_id>/shot_XXXX_img_Y.jpg   3 keyframes per shot, Y in {0,1,2}

Note: the original BaSSL keyframe URL (Aliyun OSS) is dead (HTTP 404). Frames are
sourced from the HF mirror ZhengPeng7/MovieNet, which hosts the official MovieNet data.

Usage:
  python scripts/inspect_movie.py --imdb_id tt0048545
"""

import argparse
import json
from pathlib import Path

DEFAULT_DATA_ROOT = "/mnt/disks/splice-data/datasets/movienet"


def load_movie_shots(anno_dir: Path, imdb_id: str) -> tuple[str | None, list[dict]]:
    """Return (split, shots sorted by shot index) for one movie, or (None, [])."""
    for split in ("train", "val", "test"):
        path = anno_dir / f"anno.{split}.ndjson"
        if not path.exists():
            continue
        shots = [
            rec
            for line in path.read_text().splitlines()
            if line.strip()
            for rec in [json.loads(line)]
            if rec.get("video_id") == imdb_id
        ]
        if shots:
            shots.sort(key=lambda r: int(r["shot_id"]))
            return split, shots
    return None, []


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect one MovieNet movie's annotations.")
    ap.add_argument("--imdb_id", required=True, help="IMDb id, e.g. tt0048545")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--frames_dir", default=None, help="override (default: <data_root>/240P_frames)"
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    anno_dir = data_root / "anno"
    frames_dir = Path(args.frames_dir) if args.frames_dir else data_root / "240P_frames"

    split, shots = load_movie_shots(anno_dir, args.imdb_id)
    if not shots:
        print(f"NO annotations found for {args.imdb_id} under {anno_dir}")
        return

    print(f"movie: {args.imdb_id}    split: {split}")
    print(f"shots: {len(shots)}    (num_shot field reports {shots[0].get('num_shot')})")

    # ----- scenes -----
    scene_of = {int(s["shot_id"]): s["invideo_scene_id"] for s in shots}
    bl_of = {int(s["shot_id"]): s.get("boundary_label") for s in shots}
    idxs = sorted(scene_of)
    scenes: dict[int, list[int]] = {}
    for shot_idx in idxs:
        scenes.setdefault(scene_of[shot_idx], []).append(shot_idx)
    print(f"scenes: {len(scenes)}")
    print("\nfirst 5 scenes (scene_id: shot range):")
    for sid in sorted(scenes)[:5]:
        rng = scenes[sid]
        print(f"  scene_{sid}: shots {rng[0]}-{rng[-1]}  ({len(rng)} shots)")

    # ----- adjacent cuts -----
    n_cuts = len(idxs) - 1
    n_boundary = sum(scene_of[idxs[i]] != scene_of[idxs[i + 1]] for i in range(n_cuts))
    print(
        f"\nadjacent cuts: {n_cuts}    scene-boundary cuts: {n_boundary} "
        f"({100 * n_boundary / max(n_cuts, 1):.1f}%)"
    )
    print("first 5 adjacent cuts (scene-id derived vs boundary_label of the LEFT shot):")
    for i in range(min(5, n_cuts)):
        a, b = idxs[i], idxs[i + 1]
        is_boundary = scene_of[a] != scene_of[b]
        agree = "OK" if is_boundary == bool(bl_of[a]) else "MISMATCH"
        print(
            f"  shot {a} (scene {scene_of[a]}) -> shot {b} (scene {scene_of[b]})  "
            f"scene_boundary={is_boundary}  boundary_label[{a}]={bl_of[a]}  [{agree}]"
        )

    # ----- keyframe paths -----
    print("\nkeyframe paths for first 3 shots:")
    for s in shots[:3]:
        sid = int(s["shot_id"])
        for img in (0, 1, 2):
            p = frames_dir / args.imdb_id / f"shot_{sid:04d}_img_{img}.jpg"
            print(f"  {p}   exists={p.exists()}")


if __name__ == "__main__":
    main()
