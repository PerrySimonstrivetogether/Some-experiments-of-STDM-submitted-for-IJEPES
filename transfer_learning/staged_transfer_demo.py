"""
Minimal reference implementation of two practical transfer strategies:

1. warm_start
2. freeze_then_unfreeze

This file intentionally omits the private STDM backbone and feeder-specific
data pipeline. Users should plug in their own model constructor, dataloaders,
and supervised loss.
"""

from typing import Callable, Dict

import torch
from torch.optim import Adam

from partial_transfer import transfer_compatible_parameters, freeze_parameter_subset, unfreeze_all


def supervised_epoch(model, train_loader, loss_fn: Callable, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    model.train()
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-6)
    running_loss = 0.0
    batch_count = 0
    for batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model, batch, device)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        batch_count += 1
    return running_loss / max(batch_count, 1)


def run_warm_start(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    train_loader,
    loss_fn: Callable,
    device: str = "cpu",
    lr: float = 1e-3,
) -> Dict:
    target_state, loaded_names, skipped_names = transfer_compatible_parameters(
        source_model.state_dict(), target_model.state_dict()
    )
    target_model.load_state_dict(target_state, strict=False)
    loss_value = supervised_epoch(target_model, train_loader, loss_fn, lr=lr, device=device)
    return {
        "strategy": "warm_start",
        "loaded_count": len(loaded_names),
        "skipped_count": len(skipped_names),
        "train_loss": loss_value,
    }


def run_freeze_then_unfreeze(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    train_loader,
    loss_fn: Callable,
    device: str = "cpu",
    stage1_lr: float = 1e-3,
    stage2_lr: float = 5e-4,
) -> Dict:
    target_state, loaded_names, skipped_names = transfer_compatible_parameters(
        source_model.state_dict(), target_model.state_dict()
    )
    target_model.load_state_dict(target_state, strict=False)

    freeze_parameter_subset(target_model, loaded_names)
    stage1_loss = supervised_epoch(target_model, train_loader, loss_fn, lr=stage1_lr, device=device)

    unfreeze_all(target_model)
    stage2_loss = supervised_epoch(target_model, train_loader, loss_fn, lr=stage2_lr, device=device)

    return {
        "strategy": "freeze_then_unfreeze",
        "loaded_count": len(loaded_names),
        "skipped_count": len(skipped_names),
        "stage1_loss": stage1_loss,
        "stage2_loss": stage2_loss,
    }
