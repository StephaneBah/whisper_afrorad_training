from afrorad_pipeline.layer_policy import (
    infer_whisper_size,
    resolve_unfreeze_layers,
)


def test_infer_whisper_size():
    assert infer_whisper_size("openai/whisper-base") == "base"
    assert infer_whisper_size("openai/whisper-small") == "small"
    assert infer_whisper_size("openai/whisper-medium") == "medium"


def test_resolve_unfreeze_layers_defaults():
    assert resolve_unfreeze_layers("openai/whisper-base", 6, None) == 3
    assert resolve_unfreeze_layers("openai/whisper-small", 12, None) == 4
    assert resolve_unfreeze_layers("openai/whisper-medium", 24, None) == 6


def test_resolve_unfreeze_layers_override_and_cap():
    assert resolve_unfreeze_layers("openai/whisper-small", 12, 8) == 8
    assert resolve_unfreeze_layers("openai/whisper-small", 12, 30) == 12
