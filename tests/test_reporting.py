from pathlib import Path

import pandas as pd

from afrorad_pipeline.reporting import write_eval_artifacts, write_model_comparison


def test_write_eval_artifacts_creates_csv_and_png(tmp_path):
    df = pd.DataFrame(
        [
            {
                "reference": "alpha",
                "prediction": "alpha",
                "wer": 0.0,
                "cer": 0.0,
                "sentence_accuracy": 100.0,
            },
            {
                "reference": "beta",
                "prediction": "bta",
                "wer": 50.0,
                "cer": 20.0,
                "sentence_accuracy": 0.0,
            },
        ]
    )

    artifacts = write_eval_artifacts(df=df, output_dir=str(tmp_path), model_name="test-model")

    assert Path(artifacts["per_file_csv"]).exists()
    assert Path(artifacts["summary_csv"]).exists()
    assert Path(artifacts["metrics_png"]).exists()
    assert Path(artifacts["wer_distribution_png"]).exists()


def test_write_eval_artifacts_with_prefix(tmp_path):
    df = pd.DataFrame(
        [
            {
                "reference": "x",
                "prediction": "x",
                "wer": 0.0,
                "cer": 0.0,
                "sentence_accuracy": 100.0,
            }
        ]
    )
    artifacts = write_eval_artifacts(
        df=df,
        output_dir=str(tmp_path),
        model_name="baseline-model",
        prefix="baseline",
    )

    assert Path(artifacts["per_file_csv"]).name == "baseline_predictions_per_file.csv"
    assert Path(artifacts["summary_csv"]).name == "baseline_metrics_global.csv"
    assert Path(artifacts["metrics_png"]).name == "baseline_metrics_bar.png"
    assert Path(artifacts["wer_distribution_png"]).name == "baseline_wer_distribution.png"


def test_write_model_comparison(tmp_path):
    primary = {
        "model": "fine-tuned",
        "rows": 10,
        "wer": 15.0,
        "cer": 8.0,
        "sentence_accuracy": 70.0,
    }
    baseline = {
        "model": "baseline",
        "rows": 10,
        "wer": 20.0,
        "cer": 10.0,
        "sentence_accuracy": 60.0,
    }

    csv_path = write_model_comparison(
        output_dir=str(tmp_path),
        primary_summary=primary,
        baseline_summary=baseline,
    )

    out = pd.read_csv(csv_path)
    assert len(out) == 2
    primary_row = out[out["kind"] == "primary"].iloc[0]
    assert primary_row["wer_delta_vs_baseline"] == -5.0
    assert primary_row["cer_delta_vs_baseline"] == -2.0
    assert primary_row["sentence_accuracy_delta_vs_baseline"] == 10.0
