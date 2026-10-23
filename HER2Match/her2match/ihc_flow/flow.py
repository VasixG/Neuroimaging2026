"""Linear conditional flow matching and explicit fixed-step ODE solvers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


def linear_path(
    noise: torch.Tensor, target: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    time = time[:, None, None, None]
    return (1 - time) * noise + time * target, target - noise


def flow_matching_loss(
    model: nn.Module,
    condition: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    state, velocity = linear_path(noise, target, time)
    prediction = model(state, time, condition)
    return F.mse_loss(prediction.float(), velocity.float())


@torch.inference_mode()
def integrate(
    model: nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    steps: int = 32,
    solver: str = "heun",
) -> torch.Tensor:
    if steps < 1 or solver not in NFE_PER_STEP:
        raise ValueError("Use a positive step count and euler, heun or rk4")
    if noise.shape != condition.shape:
        raise ValueError("Noise and condition shapes differ")
    state = noise.clone()
    delta = 1.0 / steps

    def velocity(image: torch.Tensor, t: float) -> torch.Tensor:
        time = torch.full((len(image),), t, device=image.device, dtype=image.dtype)
        return model(image, time, condition)

    for index in range(steps):
        t = index * delta
        k1 = velocity(state, t)
        if solver == "euler":
            state = state + delta * k1
        elif solver == "heun":
            k2 = velocity(state + delta * k1, t + delta)
            state = state + delta * (k1 + k2) / 2
        else:
            k2 = velocity(state + delta * k1 / 2, t + delta / 2)
            k3 = velocity(state + delta * k2 / 2, t + delta / 2)
            k4 = velocity(state + delta * k3, t + delta)
            state = state + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not torch.isfinite(state).all():
        raise FloatingPointError("ODE produced NaN/Inf; inspect the checkpoint and step count")
    # Keep the ODE unmodified. Clamping is only applied when rendering an RGB image.
    return state
