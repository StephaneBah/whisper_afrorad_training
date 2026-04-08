from __future__ import annotations

DEFAULT_UNFREEZE_BY_SIZE = {
    "base": 3,
    "small": 4,
    "medium": 6,
}


def infer_whisper_size(model_name: str) -> str | None:
    name = model_name.lower()
    for size in ("tiny", "base", "small", "medium", "large"):
        if f"whisper-{size}" in name:
            return size
    return None


def resolve_unfreeze_layers(
    model_name: str,
    encoder_layer_count: int,
    requested: int | None,
) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("training.encoder_unfreeze_layers must be >= 1")
        return min(requested, encoder_layer_count)

    size = infer_whisper_size(model_name)
    if size in DEFAULT_UNFREEZE_BY_SIZE:
        return min(DEFAULT_UNFREEZE_BY_SIZE[size], encoder_layer_count)

    return max(2, encoder_layer_count // 6)


def apply_encoder_freeze_policy(model, unfreeze_layers: int) -> dict[str, int]:
    for name, param in model.named_parameters():
        if name.startswith("model.encoder"):
            param.requires_grad = False
        else:
            param.requires_grad = True

    for layer_idx in range(unfreeze_layers):
        for param in model.model.encoder.layers[layer_idx].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
    }
