# v1 Feature Importance & Interpretability (Phase 3, P6)

Which parts of the 2305-d pair feature carry the signal? Feature layout: `[ eL(768) | eR(768) | |eL-eR|(768) | cos(1) ]`.

## 1. Coefficient analysis (v0 logistic head)

Coefficients are on standardised features, so magnitudes compare across slices. `mean |coef|` (per-dimension) is the fair cross-slice measure.

| slice | dims | L1 norm | L2 norm | mean \|coef\| |
|---|---|---|---|---|
| left embedding | 768 | 12.76 | 0.61 | 0.0166 |
| right embedding | 768 | 14.34 | 0.69 | 0.0187 |
| |eL - eR| | 768 | 10.02 | 0.50 | 0.0130 |
| cosine | 1 | 2.07 | 2.07 | 2.0718 |

## 2. Permutation importance

Each slice's rows are shuffled jointly (5 repeats); the AUPRC drop is its marginal importance. Baseline test AUPRC = 0.3561.

| slice | AUPRC after shuffle | AUPRC drop |
|---|---|---|
| left embedding | 0.3023 | 0.0538 |
| right embedding | 0.2591 | 0.0971 |
| |eL - eR| | 0.2663 | 0.0899 |
| cosine | 0.0782 | 0.2779 |

## 3. Ablation models (logistic re-trained on feature subsets)

| feature subset | dims | test AUPRC |
|---|---|---|
| full (2305-d) | 2305 | 0.3561 |
| concat only | 1536 | 0.1777 |
| |eL-eR| only | 768 | 0.3024 |
| cosine only | 1 | 0.2551 |
| concat + |eL-eR| | 2304 | 0.3501 |
| |eL-eR| + cosine | 769 | 0.3098 |

## 4. Linear probes -- what is in the DINOv2 embedding?

Ridge probes from frozen DINOv2 embeddings to low-level image attributes (8,000 train keyframes, 80/20 split). High R^2 means the attribute is linearly decodable from the embedding.

| attribute | probe R^2 |
|---|---|
| mean luminance | 0.676 |
| luminance contrast (std) | 0.459 |
| mean saturation | 0.366 |

## Interpretation

- **Permutation importance** ranks `cosine` first (AUPRC drop 0.2779) -- the slice the trained model leans on most.
- **Ablations**: `|eL-eR|` alone reaches AUPRC 0.302 vs the full feature's 0.356; `cosine` alone 0.255 (= the raw-cosine baseline). The difference signal carries most of the value; concat and cosine add the remainder.
- **Probes** show the DINOv2 embedding linearly encodes low-level appearance (luminance, contrast, saturation) -- exactly the cues a visual cut disturbs, which is why difference-of-embeddings is a strong continuity feature.
