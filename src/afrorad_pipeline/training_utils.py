from __future__ import annotations

from typing import Any

import torch


def build_differential_optimizer(model: Any, cfg: Any) -> torch.optim.Optimizer:
    encoder_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "model.encoder.layers" in name:
            encoder_params.append(param)
        elif "model.decoder" in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": cfg.training.encoder_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            }
        )
    if decoder_params:
        param_groups.append(
            {
                "params": decoder_params,
                "lr": cfg.training.decoder_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    return torch.optim.AdamW(param_groups)
