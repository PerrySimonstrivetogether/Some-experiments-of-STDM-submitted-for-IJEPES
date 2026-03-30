"""
Minimal unlabeled target-domain adaptation reference.

This file demonstrates:

1. self-supervised adaptation using observation consistency
2. physics-guided adaptation using an additional physics residual

The actual paper implementation used a private STDM backbone and feeder-specific
pipeline, which are not included in this public subset.
"""

from typing import Callable, Optional

import torch
from torch.optim import Adam

from adaptation_objectives import observation_consistency_loss, physics_guided_loss


def adapt_one_epoch(
    model: torch.nn.Module,
    train_loader,
    predict_fn: Callable[[torch.nn.Module, object, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str = "cpu",
    lr: float = 5e-4,
    strategy: str = "self_supervised",
    physics_operator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    lambda_phys: float = 0.1,
) -> float:
    """
    predict_fn should return:
      x_hat, x_obs, obs_mask
    """
    model.to(device)
    model.train()
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-6)

    running_loss = 0.0
    batch_count = 0
    for batch in train_loader:
        optimizer.zero_grad()
        x_hat, x_obs, obs_mask = predict_fn(model, batch, device)
        if strategy == "self_supervised":
            loss = observation_consistency_loss(x_hat, x_obs, obs_mask)
        elif strategy == "physics_guided":
            if physics_operator is None:
                raise ValueError("physics_operator must be provided for physics_guided adaptation")
            loss = physics_guided_loss(x_hat, x_obs, obs_mask, physics_operator, lambda_phys=lambda_phys)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        batch_count += 1
    return running_loss / max(batch_count, 1)
