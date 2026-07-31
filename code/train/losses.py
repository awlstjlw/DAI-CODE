"""φ-only supervised MSE loss for the learned group-selection distribution."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence with the last dimension summed out.

    JS(p ‖ q) = 0.5 * (KL(p‖m) + KL(q‖m)),   m = 0.5(p + q).
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)

    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js


def phi_loss(
    pred: dict,
    target: dict,
    eps: float = 1e-8,
    mask: torch.Tensor = None,
) -> dict:
    """Per-query MSE over the ``m`` group probabilities.

    Averages over valid groups per query before averaging across queries.

    Args:
        pred:   {"phi": (B, m)}
        target: {"phi": (B, m)}
        mask:   (B, m) bool, True for valid groups.

    Returns:
        dict with keys: total, l_phi
    """
    phi_hat = pred["phi"]
    phi_star = target["phi"]

    if mask is not None:
        # Re-normalize predicted probabilities over valid groups only.
        valid_sum = (phi_hat * mask.float()).sum(dim=-1, keepdim=True).clamp(min=eps)
        phi_hat = phi_hat * mask.float() / valid_sum

    squared_error = (phi_hat - phi_star).pow(2)
    if mask is not None:
        mask_f = mask.float()
        valid_groups_per_query = mask_f.sum(dim=-1).clamp(min=1.0)
        per_query_loss = (
            (squared_error * mask_f).sum(dim=-1) / valid_groups_per_query
        )
    else:
        per_query_loss = squared_error.mean(dim=-1)
    l_phi = per_query_loss.mean()

    return {
        "total": l_phi,
        "l_phi": l_phi.detach(),
    }


def fair_pivot_loss(
    pred: dict,
    target: dict,
    lambda_phi: float = 1.0,
    lambda_V: float = 1.0,
    lambda_p: float = 1.0,
    eps: float = 1e-8,
    mask: torch.Tensor = None,
) -> dict:
    """Backward-compatible wrapper that delegates to phi_loss.

    The lambda_* arguments are accepted but unused to avoid breaking callers.
    """
    return phi_loss(pred, target, eps=eps, mask=mask)
