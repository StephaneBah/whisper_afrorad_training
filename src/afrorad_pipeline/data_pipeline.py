from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, load_dataset


def _normalize_relpath(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    normalized = normalized.lstrip("./")
    normalized = normalized.lstrip("/")
    return normalized


def _extract_manifest_paths(manifest_data: dict[str, Any]) -> set[str]:
    paths: set[str] = set()

    for item in manifest_data.get("audios", []):
        if isinstance(item, str):
            norm = _normalize_relpath(item)
            if norm:
                paths.add(norm)

    for item in manifest_data.get("entries", []):
        if isinstance(item, str):
            norm = _normalize_relpath(item)
            if norm:
                paths.add(norm)
            continue

        if isinstance(item, dict):
            candidate = item.get("audio") or item.get("path")
            if isinstance(candidate, str):
                norm = _normalize_relpath(candidate)
                if norm:
                    paths.add(norm)

    return paths


def _is_manifest_match(sample_path: str, allowed_paths: set[str]) -> bool:
    sample = _normalize_relpath(sample_path)
    if not sample:
        return False

    for allowed in allowed_paths:
        if sample == allowed or sample.endswith(f"/{allowed}"):
            return True
    return False


def load_afrorad_dataset(cfg: Any, token: str | None) -> DatasetDict:
    kwargs = {
        "path": cfg.data.hf_dataset_id,
        "streaming": cfg.data.streaming,
    }
    if cfg.data.dataset_config_name:
        kwargs["name"] = cfg.data.dataset_config_name
    if cfg.data.use_auth_token:
        kwargs["token"] = token

    dataset = load_dataset(**kwargs)
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError("Expected DatasetDict with train/test splits.")
    return dataset


def maybe_filter_with_manifest(
    dataset: Dataset,
    audio_column: str,
    manifest_path: str,
    fail_on_missing: bool,
) -> Dataset:
    path = Path(manifest_path)
    if not path.exists():
        if fail_on_missing:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        return dataset

    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = _extract_manifest_paths(data)
    if not allowed:
        return dataset

    def _is_allowed(sample: dict) -> bool:
        audio_entry = sample.get(audio_column)
        if isinstance(audio_entry, dict):
            sample_path = audio_entry.get("path", "")
        else:
            sample_path = str(audio_entry)

        return _is_manifest_match(str(sample_path), allowed)

    return dataset.filter(_is_allowed)


def normalize_text(text: str, lower: bool, strip_newlines: bool) -> str:
    out = text
    if strip_newlines:
        out = out.replace("\n", " ")
    if lower:
        out = out.lower()
    return out.strip()


def build_preprocess_fn(cfg: Any, processor: Any):
    audio_col = cfg.data.audio_column
    text_col = cfg.data.text_column

    def _prepare(batch: dict) -> dict:
        audio = batch[audio_col]
        text = normalize_text(
            batch[text_col],
            lower=cfg.preprocessing.lowercase,
            strip_newlines=cfg.preprocessing.strip_newlines,
        )

        batch_out = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        )

        labels = processor.tokenizer(
            text,
            truncation=True,
            max_length=cfg.training.max_label_length,
        ).input_ids

        return {
            "input_features": batch_out["input_features"][0],
            "labels": labels,
        }

    return _prepare


def cast_audio_column(dataset: Dataset, audio_column: str, sample_rate: int) -> Dataset:
    return dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))
