from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x, mask):
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def temporal_huber_adapted(pred_disp, target_disp, frame_mask, target_player_mask):
    """Main loss: time-decayed Huber + trajectory smoothness penalties."""
    mask = frame_mask * target_player_mask[:, None, :]
    huber = F.smooth_l1_loss(pred_disp, target_disp, reduction="none", beta=1.0)
    t = torch.arange(pred_disp.size(1), device=pred_disp.device, dtype=pred_disp.dtype)
    weights = torch.exp(-0.03 * t).view(1, -1, 1, 1)
    main = _masked_mean(huber * weights, mask)

    # Work in predicted absolute coordinates relative to the final observed point.
    velocity = pred_disp[:, 1:] - pred_disp[:, :-1]
    velocity_mask = mask[:, 1:] * mask[:, :-1]
    velocity_loss = _masked_mean(velocity.abs(), velocity_mask)

    if pred_disp.size(1) >= 3:
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        accel_mask = velocity_mask[:, 1:] * velocity_mask[:, :-1]
        accel_loss = _masked_mean(acceleration.abs(), accel_mask)
    else:
        accel_loss = pred_disp.new_tensor(0.0)

    return main + 0.01 * velocity_loss + 0.005 * accel_loss, {
        "main_huber": main.detach(),
        "velocity_smooth": velocity_loss.detach(),
        "acceleration_smooth": accel_loss.detach(),
    }


def total_loss(outputs, target_disp, target_frame_mask, x_hist, time_mask, player_mask, target_mask):
    main, info = temporal_huber_adapted(
        outputs["main"], target_disp, target_frame_mask, target_mask
    )

    # Auxiliary 1: next-frame displacement inside the observed history.
    obs_delta = x_hist[:, 1:, :, :2] - x_hist[:, :-1, :, :2]
    inter_mask = (
        time_mask[:, 1:] * time_mask[:, :-1]
    )[:, :, None] * player_mask[:, None, :]
    inter = _masked_mean(
        F.smooth_l1_loss(outputs["inter"], obs_delta, reduction="none"), inter_mask
    )

    # Auxiliary 2: displacement from every observed frame to each player's
    # own final valid future target. Horizons are player-specific in the
    # competition, so using target_disp[:, -1] would incorrectly treat padded
    # values as real endpoints for shorter trajectories.
    B, H, P, _ = target_disp.shape
    valid_counts = target_frame_mask.sum(dim=1).long()
    last_idx = (valid_counts - 1).clamp_min(0)
    gather_idx = last_idx.unsqueeze(1).unsqueeze(-1).expand(B, 1, P, 2)
    final_disp = target_disp.gather(1, gather_idx).squeeze(1)
    final_abs_target = x_hist[:, -1, :, :2] + final_disp
    endpoint_target = final_abs_target[:, None, :, :] - x_hist[:, :, :, :2]
    endpoint_mask = time_mask[:, :, None] * target_mask[:, None, :]
    endpoint = _masked_mean(
        F.smooth_l1_loss(outputs["endpoint"], endpoint_target, reduction="none"),
        endpoint_mask,
    )

    total = main + inter + 0.1 * endpoint
    info.update({"main": main.detach(), "inter": inter.detach(), "endpoint": endpoint.detach()})
    return total, info
