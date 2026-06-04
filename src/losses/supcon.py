# Supervised Contrastive Loss (Khosla et al. 2020).
# for each anchor i, log p(positive | i) over all other batch elements,
# averaged over the set of same-class positives P(i). anchors with no positive
# in the batch are skipped (they contribute no useful gradient).

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    # features: (N, D), L2-normalised. labels: (N,) integer.
    def forward(self, features, labels):
        N = features.size(0)
        device = features.device

        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye

        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return (features * 0).sum()  # zero scalar, stays in compute graph

        # log-denominator over j != i (logsumexp for numerical stability)
        log_denom = torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True)
        log_prob = sim - log_denom

        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
        return per_anchor[has_pos].mean()
