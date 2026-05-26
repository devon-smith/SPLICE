"""v2: LoRA fine-tuning of DINOv2 attention layers + MLP head.

Adds LoRA adapters to DINOv2's attention query and value projections (all 12
layers of ViT-B/14). All other backbone params remain frozen. The pair feature
and MLP head are identical to v1, but built differentiably so gradients flow
through the backbone. Backbone and head use separate learning rates.

Single run (overriding config defaults):
  python scripts/train/v2_lora.py \\
      --cut_index /mnt/disks/splice-data/outputs/cut_index/cuts.parquet \\
      --out /mnt/disks/splice-data/outputs/v2_lora \\
      --r 8 --lora_alpha 16

Sweep (all r x lora_alpha x lr_backbone combos from configs/v2_lora.yaml):
  python scripts/train/v2_lora.py --sweep \\
      --cut_index /mnt/disks/splice-data/outputs/cut_index/cuts.parquet \\
      --out /mnt/disks/splice-data/outputs/v2_lora_sweep
"""

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics  # noqa: E402

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise SystemExit("peft not installed — run: pip install peft")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2_lora")

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "v2_lora.yaml"
DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/v2_lora"

# DINOv2 uses ImageNet stats
_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CutDataset(Dataset):
    """Loads boundary frame pairs on-the-fly from the cut index."""

    def __init__(self, cuts: pd.DataFrame) -> None:
        self.left = cuts["left_img2_path"].to_numpy()
        self.right = cuts["right_img0_path"].to_numpy()
        self.labels = cuts["y_inconsistent"].to_numpy().astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        pv_l = _TRANSFORM(Image.open(self.left[i]).convert("RGB"))
        pv_r = _TRANSFORM(Image.open(self.right[i]).convert("RGB"))
        return pv_l, pv_r, torch.tensor(self.labels[i])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PairMLP(nn.Module):
    """[eL | eR | |eL-eR| | cos(eL,eR)] -> scalar logit. Same arch as v1."""

    def __init__(self, emb_dim: int = 768, hidden=(512, 128), dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = 2 * emb_dim + emb_dim + 1  # 2305
        layers: list[nn.Module] = []
        dim = in_dim
        for w in hidden:
            layers += [nn.Linear(dim, w), nn.ReLU(), nn.Dropout(dropout)]
            dim = w
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, e_l: torch.Tensor, e_r: torch.Tensor) -> torch.Tensor:
        abs_diff = (e_l - e_r).abs()
        cos = F.cosine_similarity(e_l, e_r, eps=1e-8).unsqueeze(-1)
        return self.net(torch.cat([e_l, e_r, abs_diff, cos], dim=-1)).squeeze(-1)


def _build_backbone(model_id: str, r: int, lora_alpha: int, target_modules: list, lora_dropout: float):
    base = AutoModel.from_pretrained(model_id)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    return get_peft_model(base, lora_cfg)


def _encode_pairs(
    backbone, pv_left: torch.Tensor, pv_right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One backbone forward pass for both frames; returns (e_left, e_right) float32."""
    B = pv_left.size(0)
    pv = torch.cat([pv_left, pv_right], dim=0)
    out = backbone(pixel_values=pv)
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0]
    emb = pooled.float()
    return emb[:B], emb[B:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _sample_epoch(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n >= len(df):
        return df
    return df.iloc[rng.choice(len(df), n, replace=False)].reset_index(drop=True)


def train_one(cut_index: str, out_dir: Path, cfg: dict, run_name: str,
              wandb_project: str, wandb_mode: str, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    on_gpu = device == "cuda"

    df = pd.read_parquet(cut_index, columns=[
        "split", "y_inconsistent", "left_img2_path", "right_img0_path"
    ])
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    log.info("train %d | val %d | test %d | pos %.2f%%",
             len(splits["train"]), len(splits["val"]), len(splits["test"]),
             100 * df["y_inconsistent"].mean())

    lora_cfg = cfg["lora"]
    backbone = _build_backbone(
        model_id=cfg.get("backbone", "facebook/dinov2-base"),
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
    ).to(device)
    backbone.print_trainable_parameters()

    head_cfg = cfg.get("head", {})
    head = PairMLP(
        emb_dim=768,
        hidden=tuple(head_cfg.get("hidden", [512, 128])),
        dropout=head_cfg.get("dropout", 0.1),
    ).to(device)

    pos_weight = torch.tensor(
        [(splits["train"]["y_inconsistent"] == 0).sum() /
         max((splits["train"]["y_inconsistent"] == 1).sum(), 1)],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info("pos_weight=%.2f", pos_weight.item())

    optim_cfg = cfg.get("optim", {})
    optim = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": optim_cfg.get("lr_backbone", 5e-5)},
        {"params": head.parameters(), "lr": optim_cfg.get("lr_head", 1e-3)},
    ], weight_decay=optim_cfg.get("weight_decay", 1e-4))

    train_cfg = cfg.get("train", {})
    epochs = train_cfg.get("epochs", 10)
    patience = train_cfg.get("patience", 3)
    batch_size = train_cfg.get("batch_size", 32)
    max_per_epoch = train_cfg.get("max_cuts_per_epoch", 25000)
    grad_accum = train_cfg.get("grad_accumulation", 4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = GradScaler(enabled=on_gpu)
    run = _open_wandb(wandb_project, wandb_mode, run_name, {**cfg, "seed": seed})

    best_auprc, best_bb_state, best_head_state, best_epoch, stale = -1.0, None, None, -1, 0

    for epoch in range(epochs):
        backbone.train()
        head.train()
        epoch_df = _sample_epoch(splits["train"], max_per_epoch, rng)
        loader = DataLoader(
            CutDataset(epoch_df),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=on_gpu,
            drop_last=False,
        )
        running, n_seen = 0.0, 0
        optim.zero_grad()
        for step, (pv_l, pv_r, y) in enumerate(loader):
            pv_l = pv_l.to(device, non_blocking=True)
            pv_r = pv_r.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device, enabled=on_gpu):
                e_l, e_r = _encode_pairs(backbone, pv_l, pv_r)
                logit = head(e_l.float(), e_r.float())
                loss = criterion(logit, y) / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            running += loss.item() * grad_accum * len(y)
            n_seen += len(y)
        sched.step()

        backbone.eval()
        head.eval()
        val_scores = _score_split(backbone, head, splits["val"], batch_size * 2, device, on_gpu)
        val_auprc = ranking_metrics(splits["val"]["y_inconsistent"].to_numpy().astype(int), val_scores)["auprc"]
        log.info("epoch %2d  loss %.4f  val_auprc %.4f  (best %.4f ep %d)",
                 epoch, running / n_seen, val_auprc, best_auprc, best_epoch)
        _wandb_log(run, {"epoch": epoch, "train_loss": running / n_seen, "val_auprc": val_auprc,
                         "lr_backbone": sched.get_last_lr()[0]})

        if val_auprc > best_auprc:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            best_bb_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
            best_head_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best ep %d)", epoch, best_epoch)
                break

    backbone.load_state_dict(best_bb_state)
    head.load_state_dict(best_head_state)
    backbone.eval()
    head.eval()

    val_y = splits["val"]["y_inconsistent"].to_numpy().astype(int)
    test_y = splits["test"]["y_inconsistent"].to_numpy().astype(int)
    val_scores = _score_split(backbone, head, splits["val"], batch_size * 2, device, on_gpu)
    test_scores = _score_split(backbone, head, splits["test"], batch_size * 2, device, on_gpu)
    thr = best_f1_threshold(val_y, val_scores)
    rank = ranking_metrics(test_y, test_scores)
    hard = threshold_metrics(test_y, test_scores, thr)

    result = {
        "run_name": run_name,
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "lr_backbone": optim_cfg.get("lr_backbone", 5e-5),
        "lr_head": optim_cfg.get("lr_head", 1e-3),
        "best_epoch": best_epoch,
        "val_auprc": float(best_auprc),
        "test_auroc": rank["auroc"],
        "test_auprc": rank["auprc"],
        "f1_at_val_thr": hard["f1"],
        "precision_at_val_thr": hard["precision"],
        "recall_at_val_thr": hard["recall"],
        "val_thr": float(thr),
        "n_test": rank["n"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(run_dir / "backbone_lora")   # saves only adapter weights
    torch.save(best_head_state, run_dir / "head.pt")
    np.savez(run_dir / "scores.npz", val_s=val_scores, val_y=val_y, test_s=test_scores, test_y=test_y)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    _wandb_log(run, {f"test_{k}": v for k, v in rank.items() if isinstance(v, float)})
    if run is not None:
        try:
            run.finish()
        except Exception:  # noqa: BLE001
            pass

    return result


@torch.inference_mode()
def _score_split(backbone, head, split_df: pd.DataFrame,
                 batch_size: int, device: str, on_gpu: bool) -> np.ndarray:
    loader = DataLoader(CutDataset(split_df), batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=on_gpu)
    parts = []
    for pv_l, pv_r, _ in loader:
        pv_l = pv_l.to(device)
        pv_r = pv_r.to(device)
        with torch.autocast(device_type=device, enabled=on_gpu):
            e_l, e_r = _encode_pairs(backbone, pv_l, pv_r)
        s = torch.sigmoid(head(e_l.float(), e_r.float()))
        parts.append(s.cpu().numpy())
    return np.concatenate(parts)


def _open_wandb(project, mode, name, config):
    if mode == "disabled":
        return None
    try:
        import os
        import wandb
        has_key = bool(os.environ.get("WANDB_API_KEY")) or bool(wandb.api.api_key)
        resolved = mode if (mode != "online" or has_key) else "offline"
        return wandb.init(project=project, name=name, mode=resolved, config=config)
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B init failed: %s", exc)
        return None


def _wandb_log(run, payload):
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# main / sweep
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--sweep", action="store_true",
                    help="run all r x lora_alpha x lr_backbone combos from the config")
    # single-run overrides
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--lr_backbone", type=float, default=None)
    ap.add_argument("--lr_head", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=231)
    ap.add_argument("--wandb_project", default="splice-v2-lora")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.sweep:
        grid = cfg.get("sweep", {})
        rs = grid.get("r", [cfg["lora"]["r"]])
        alphas = grid.get("lora_alpha", [cfg["lora"]["lora_alpha"]])
        lr_bbs = grid.get("lr_backbone", [cfg["optim"].get("lr_backbone", 5e-5)])
        all_results = []
        for r in rs:
            for alpha in alphas:
                for lr_bb in lr_bbs:
                    run_cfg = copy.deepcopy(cfg)
                    run_cfg["lora"]["r"] = r
                    run_cfg["lora"]["lora_alpha"] = alpha
                    run_cfg["optim"]["lr_backbone"] = lr_bb
                    run_name = f"r{r}_a{alpha}_lrbb{lr_bb:.0e}"
                    log.info("=== sweep: %s ===", run_name)
                    result = train_one(args.cut_index, out_dir, run_cfg, run_name,
                                       args.wandb_project, args.wandb_mode, args.seed)
                    all_results.append(result)
                    _print_result(result)
        print("\n=== sweep summary (sorted by test AUPRC) ===")
        for row in sorted(all_results, key=lambda x: -x["test_auprc"]):
            print(f"  {row['run_name']:35s}  val {row['val_auprc']:.4f}  test {row['test_auprc']:.4f}")
        (out_dir / "sweep_results.json").write_text(json.dumps(all_results, indent=2))
    else:
        if args.r is not None:
            cfg["lora"]["r"] = args.r
        if args.lora_alpha is not None:
            cfg["lora"]["lora_alpha"] = args.lora_alpha
        if args.lr_backbone is not None:
            cfg["optim"]["lr_backbone"] = args.lr_backbone
        if args.lr_head is not None:
            cfg["optim"]["lr_head"] = args.lr_head
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        run_name = f"r{cfg['lora']['r']}_a{cfg['lora']['lora_alpha']}"
        result = train_one(args.cut_index, out_dir, cfg, run_name,
                           args.wandb_project, args.wandb_mode, args.seed)
        _print_result(result)


def _print_result(r: dict) -> None:
    print(f"\n=== {r['run_name']} ===")
    for k in ("test_auprc", "test_auroc", "f1_at_val_thr", "val_auprc"):
        print(f"  {k:22s}  {r[k]:.4f}")
    print(f"  best epoch  {r['best_epoch']}  (v1 baseline: 0.409)")


if __name__ == "__main__":
    main()
