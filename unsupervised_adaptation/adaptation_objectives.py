from typing import Callable

import torch


def observation_consistency_loss(x_hat: torch.Tensor, x_obs: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    residual = (x_hat - x_obs) * obs_mask
    return (residual ** 2).sum() / obs_mask.sum().clamp_min(1.0)


def physics_guided_loss(
    x_hat: torch.Tensor,
    x_obs: torch.Tensor,
    obs_mask: torch.Tensor,
    physics_operator: Callable[[torch.Tensor], torch.Tensor],
    lambda_phys: float = 0.1,
) -> torch.Tensor:
    obs_loss = observation_consistency_loss(x_hat, x_obs, obs_mask)
    z_hat = physics_operator(x_hat)
    z_obs = physics_operator(x_obs)
    phys_loss = ((z_hat - z_obs) ** 2).mean()
    return obs_loss + lambda_phys * phys_loss
