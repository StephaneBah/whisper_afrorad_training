import json

import pytest
from datasets import Dataset

from afrorad_pipeline.data_pipeline import maybe_filter_with_manifest


def test_manifest_filter_missing_file_returns_input(tmp_path):
    ds = Dataset.from_list([
        {"audio": {"path": "a/report_1.wav"}, "text": "x"},
        {"audio": {"path": "a/report_2.wav"}, "text": "y"},
    ])
    out = maybe_filter_with_manifest(
        ds,
        audio_column="audio",
        manifest_path=str(tmp_path / "missing.json"),
        fail_on_missing=False,
    )
    assert len(out) == 2


def test_manifest_filter_legacy_audios_list(tmp_path):
    ds = Dataset.from_list(
        [
            {"audio": {"path": "root/report_1/recorded_audio.wav"}, "text": "x"},
            {"audio": {"path": "root/report_2/recorded_audio.wav"}, "text": "y"},
        ]
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"audios": ["report_2/recorded_audio.wav"]}),
        encoding="utf-8",
    )

    out = maybe_filter_with_manifest(
        ds,
        audio_column="audio",
        manifest_path=str(manifest),
        fail_on_missing=False,
    )
    assert len(out) == 1
    assert out[0]["audio"]["path"].endswith("report_2/recorded_audio.wav")


def test_manifest_filter_entries_schema(tmp_path):
    ds = Dataset.from_list(
        [
            {"audio": {"path": "report_a/recorded_audio.wav"}, "text": "a"},
            {"audio": {"path": "report_b/recorded_audio.wav"}, "text": "b"},
        ]
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "entries": [
                    {"id": "b", "audio": "report_b/recorded_audio.wav", "split": "test"}
                ],
            }
        ),
        encoding="utf-8",
    )

    out = maybe_filter_with_manifest(
        ds,
        audio_column="audio",
        manifest_path=str(manifest),
        fail_on_missing=False,
    )
    assert len(out) == 1
    assert out[0]["audio"]["path"] == "report_b/recorded_audio.wav"


def test_manifest_filter_missing_file_raises_when_requested(tmp_path):
    ds = Dataset.from_list([{"audio": {"path": "a.wav"}, "text": "x"}])

    with pytest.raises(FileNotFoundError):
        maybe_filter_with_manifest(
            ds,
            audio_column="audio",
            manifest_path=str(tmp_path / "missing.json"),
            fail_on_missing=True,
        )
