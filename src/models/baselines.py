# non-learned baselines for cut-continuity scoring
# all map a cut to a scalar where HIGHER means MORE inconsistent
# (HSV chi-square histogram distance, 1 - CLIP cosine; raw DINOv2 cosine is read
# straight off the cached pair feature in v0_logistic.py)

from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPVisionModelWithProjection


# --- HSV histogram chi-square ----------------------------------------------

def hsv_histogram(path, bins=(8, 8, 8)):
    img = cv2.imread(str(path))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


# symmetric chi-square distance between two histograms (0 = identical)
def chi_square(h1, h2):
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10)))


def _hist_worker(arg):
    path, bins = arg
    return hsv_histogram(path, bins)


# parallel HSV histograms for a list of image paths -> {path: histogram}
def compute_hsv_histograms(paths, bins=(8, 8, 8), n_workers=12):
    paths = list(paths)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        hists = list(pool.map(_hist_worker, [(p, bins) for p in paths], chunksize=256))
    return {p: h for p, h in zip(paths, hists) if h is not None}


# --- CLIP cosine ------------------------------------------------------------

class _PathImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return self.paths[i], pv


class CLIPImageEncoder:
    def __init__(self, model_id="openai/clip-vit-large-patch14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        # vision tower + projection head; .image_embeds is the joint-space embedding
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def encode_paths(self, paths, batch_size=128, num_workers=8):
        paths = list(paths)
        loader = DataLoader(_PathImageDataset(paths, self.processor),
                            batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        out = {}
        on_gpu = self.device == "cuda"
        for batch_paths, pixel_values in loader:
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda" if on_gpu else "cpu", enabled=on_gpu):
                vis_out = self.model(pixel_values=pixel_values)
            feat = torch.nn.functional.normalize(vis_out.image_embeds.float(), dim=-1).cpu().numpy()
            for path, vec in zip(batch_paths, feat):
                out[path] = vec
        return out


# per-pair 1 - cos (higher = more inconsistent); NaN if either embedding is missing
def cosine_distance_scores(emb, left_paths, right_paths):
    scores = np.full(len(left_paths), np.nan, dtype=np.float64)
    for i, (lp, rp) in enumerate(zip(left_paths, right_paths)):
        el, er = emb.get(lp), emb.get(rp)
        if el is None or er is None:
            continue
        denom = (np.linalg.norm(el) * np.linalg.norm(er)) + 1e-10
        scores[i] = 1.0 - float(np.dot(el, er) / denom)
    return scores


# per-pair HSV chi-square distance; NaN if either histogram is missing
def chisq_scores(hists, left_paths, right_paths):
    scores = np.full(len(left_paths), np.nan, dtype=np.float64)
    for i, (lp, rp) in enumerate(zip(left_paths, right_paths)):
        hl, hr = hists.get(lp), hists.get(rp)
        if hl is None or hr is None:
            continue
        scores[i] = chi_square(hl, hr)
    return scores
