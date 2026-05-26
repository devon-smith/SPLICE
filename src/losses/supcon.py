"""Supervised Contrastive Loss (Khosla et al., 2020).

For a batch of L2-normalised embeddings z_i with integer labels y_i:

    L = (1/N) Σ_i  (-1/|P(i)|) Σ_{p∈P(i)}  log ──────────────────────────────
                                                  Σ_{a≠i} exp(z_i·z_a / τ)

where P(i) = {j ≠ i : y_j = y_i} is the set of same-class positives.

Anchors with no positive in the batch are excluded from the mean rather than
contributing a zero (they add no useful gradient signal).
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (N, D) -- must be L2-normalised before calling
        labels   : (N,)   -- integer class labels
        Returns a scalar loss; zero (in-graph) if no anchor has a positive.
        """
        N = features.size(0)
        device = features.device

        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        eye = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye  # (N, N)

        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return (features * 0).sum()  # zero scalar, stays in compute graph

        # Log-denominator: sum over all j ≠ i (logsumexp for stability)
        log_denom = torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True)
        log_prob = sim - log_denom  # (N, N): log p(j | i)

        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos

        return per_anchor[has_pos].mean()
