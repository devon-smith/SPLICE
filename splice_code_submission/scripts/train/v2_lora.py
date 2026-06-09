# AI-USE: This script was AI-assisted with Claude via Claude Code.
# trains the LoRA adapter + head jointly (rest of DINOv2 frozen)


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics

CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
OUT_BASE = "/mnt/disks/splice-data/outputs/v2_lora"

# hyperparams (the "v2 final" config from reports/v2_final.md)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LR_BACKBONE = 5e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 35
PATIENCE = 7
BATCH_SIZE = 32
GRAD_ACCUM = 4
MAX_CUTS_PER_EPOCH = 25000

# CS231N Lec 5: Image resizing and normalization preprocessing
# DINOv2 uses ImageNet stats
tfm = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CutDataset(Dataset):
    def __init__(self, cuts):
        self.left = cuts["left_img2_path"].to_numpy()
        self.right = cuts["right_img0_path"].to_numpy()
        self.labels = cuts["y_inconsistent"].to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pv_l = tfm(Image.open(self.left[i]).convert("RGB"))
        pv_r = tfm(Image.open(self.right[i]).convert("RGB"))
        return pv_l, pv_r, torch.tensor(self.labels[i])


# CS231N Lec 4: MLP forward pass (2305 -> 512 -> 128 -> 1)
class PairMLP(nn.Module):
    def __init__(self, emb_dim=768, hidden=(512, 128), dropout=0.1):
        super().__init__()
        in_dim = 2 * emb_dim + emb_dim + 1  # 2305
        self.norm = nn.LayerNorm(in_dim)  # CS231N Lec 6: LayerNorm in PairMLP head
        layers: list[nn.Module] = []
        dim = in_dim
        for w in hidden:
            layers += [nn.Linear(dim, w), nn.ReLU(), nn.Dropout(dropout)]  # CS231N Lec 4: ReLU activations; Dropout
            dim = w
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, e_l: torch.Tensor, e_r: torch.Tensor) -> torch.Tensor:
        # CS231N Lec 4: Backpropagation through the pair feature head
        abs_diff = (e_l - e_r).abs()
        cos = F.cosine_similarity(e_l, e_r, eps=1e-8).unsqueeze(-1)
        x = torch.cat([e_l, e_r, abs_diff, cos], dim=-1)
        return self.net(self.norm(x)).squeeze(-1)


# one backbone forward pass for both frames; returns (e_left, e_right) float32
def encode_pairs(backbone, pv_left, pv_right):
    b = pv_left.size(0)
    pv = torch.cat([pv_left, pv_right], dim=0)
    out = backbone(pixel_values=pv)  # CS231N Lec 8: Vision Transformer (ViT-B/14) backbone — patch token embeddings
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0]  # CS231N Lec 8: CLS token as global image representation
    emb = pooled.float()
    return emb[:b], emb[b:]


@torch.inference_mode()
def score_split(backbone, head, split_df, batch_size, device, on_gpu):
    loader = DataLoader(CutDataset(split_df), batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=on_gpu)
    parts = []
    for pv_l, pv_r, _ in loader:
        pv_l, pv_r = pv_l.to(device), pv_r.to(device)
        with torch.autocast(device_type=device, enabled=on_gpu):
            e_l, e_r = encode_pairs(backbone, pv_l, pv_r)
        s = torch.sigmoid(head(e_l.float(), e_r.float()))
        parts.append(s.cpu().numpy())
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_BASE)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    on_gpu = device == "cuda"

    df = pd.read_parquet(CUT_INDEX, columns=["split", "y_inconsistent",
                                              "left_img2_path", "right_img0_path"])
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    print(f"train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])} | "
          f"pos {100 * df['y_inconsistent'].mean():.2f}%")

    # build LoRA-wrapped backbone (only q,v adapters trainable; everything else frozen)
    base = AutoModel.from_pretrained("facebook/dinov2-base")  # CS231N Lec 6: Transfer learning: loading pretrained DINOv2 weights
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,  # CS231N Lec 8: LoRA: low-rank adapter matrices on query and value projections
                          target_modules=["query", "value"],  # CS231N Lec 8: Self-attention layers targeted by LoRA adapters
                          lora_dropout=LORA_DROPOUT, bias="none")
    backbone = get_peft_model(base, lora_cfg).to(device)  # CS231N Lec 6: Fine-tuning pretrained backbone
    backbone.print_trainable_parameters()

    head = PairMLP(emb_dim=768, hidden=(512, 128), dropout=0.1).to(device)

    # class-weighted BCE since positives are ~7.5% of train
    pos_weight = torch.tensor(
        [(splits["train"]["y_inconsistent"] == 0).sum() /
         max((splits["train"]["y_inconsistent"] == 1).sum(), 1)],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # CS231N Lec 3: Class-weighted BCE loss for 7.47% positive rate imbalance
    print(f"pos_weight={pos_weight.item():.2f}")

    optim = torch.optim.AdamW([  # CS231N Lec 3: Adam optimizer usage
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)  # CS231N Lec 3: Weight decay / L2 regularization on MLP head
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)  # CS231N Lec 3: Learning rate schedules (cosine LR schedule in v2 LoRA training)
    scaler = GradScaler(enabled=on_gpu)  # CS231N Lec 11: fp16 mixed precision training

    out_dir = Path(args.out) / f"r{LORA_R}_a{LORA_ALPHA}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auprc, best_epoch, stale = -1.0, -1, 0
    best_lora_state, best_head_state = None, None

    for epoch in range(EPOCHS):
        backbone.train()
        head.train()
        # subsample 25k cuts per epoch for fast iteration
        train_df = splits["train"].iloc[
            rng.choice(len(splits["train"]), MAX_CUTS_PER_EPOCH, replace=False)
        ].reset_index(drop=True) if len(splits["train"]) > MAX_CUTS_PER_EPOCH else splits["train"]
        loader = DataLoader(CutDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=on_gpu)

        running, n_seen = 0.0, 0
        optim.zero_grad()
        stepped_at_end = False
        for step, (pv_l, pv_r, y) in enumerate(loader):
            pv_l = pv_l.to(device, non_blocking=True)
            pv_r = pv_r.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=on_gpu):
                e_l, e_r = encode_pairs(backbone, pv_l, pv_r)
                logit = head(e_l.float(), e_r.float())
                loss = criterion(logit, y) / GRAD_ACCUM
            scaler.scale(loss).backward()  # CS231N Lec 11: Gradient accumulation
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                stepped_at_end = True
            else:
                stepped_at_end = False
            running += loss.item() * GRAD_ACCUM * len(y)
            n_seen += len(y)
        # flush any leftover partial-accum batch
        if n_seen and not stepped_at_end:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
        sched.step()

        backbone.eval()
        head.eval()
        val_scores = score_split(backbone, head, splits["val"], BATCH_SIZE * 2, device, on_gpu)
        val_y = splits["val"]["y_inconsistent"].to_numpy().astype(int)
        val_auprc = ranking_metrics(val_y, val_scores)["auprc"]
        print(f"epoch {epoch:2d}  loss {running/n_seen:.4f}  val_auprc {val_auprc:.4f}  "
              f"(best {best_auprc:.4f} ep {best_epoch})")

        if val_auprc > best_auprc:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            best_lora_state = {k: v.detach().cpu().clone()  # CS231N Lec 11: Checkpoint saving and resume
                               for k, v in get_peft_model_state_dict(backbone).items()}
            best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"early stop at epoch {epoch} (best ep {best_epoch})")
            break

    # restore best-by-val checkpoint and score val + test once
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(backbone, best_lora_state)
    head.load_state_dict(best_head_state)
    backbone.eval()
    head.eval()

    val_y = splits["val"]["y_inconsistent"].to_numpy().astype(int)
    test_y = splits["test"]["y_inconsistent"].to_numpy().astype(int)
    val_scores = score_split(backbone, head, splits["val"], BATCH_SIZE * 2, device, on_gpu)
    test_scores = score_split(backbone, head, splits["test"], BATCH_SIZE * 2, device, on_gpu)
    thr = best_f1_threshold(val_y, val_scores)
    rank = ranking_metrics(test_y, test_scores)
    hard = threshold_metrics(test_y, test_scores, thr)

    result = {
        "seed": args.seed,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "best_epoch": best_epoch,
        "val_auprc": float(best_auprc),
        "test_auroc": rank["auroc"],
        "test_auprc": rank["auprc"],
        "f1_at_val_thr": hard["f1"],
        "precision_at_val_thr": hard["precision"],
        "recall_at_val_thr": hard["recall"],
        "val_thr": float(thr),
        "n_test": rank["n"],
    }
    backbone.save_pretrained(out_dir / "backbone_lora")  # adapter weights only
    torch.save(best_head_state, out_dir / "head.pt")
    np.savez(out_dir / "scores.npz", val_s=val_scores, val_y=val_y,
             test_s=test_scores, test_y=test_y)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))

    print(f"\ntest_auprc {rank['auprc']:.4f}  test_auroc {rank['auroc']:.4f}  "
          f"f1 {hard['f1']:.4f}  best_epoch {best_epoch}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
