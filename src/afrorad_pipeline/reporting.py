from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import ensure_dir


def build_eval_summary(df: pd.DataFrame, model_name: str) -> dict[str, float | int | str]:
    return {
        "model": model_name,
        "rows": int(len(df)),
        "wer": float(df["wer"].mean()) if len(df) else 0.0,
        "cer": float(df["cer"].mean()) if len(df) else 0.0,
        "sentence_accuracy": float(df["sentence_accuracy"].mean()) if len(df) else 0.0,
    }


def _prefixed_name(prefix: str | None, base_name: str) -> str:
    if prefix:
        return f"{prefix}_{base_name}"
    return base_name


def _save_metrics_bar(summary: dict[str, float | int | str], out_dir: Path) -> Path:
    metrics = {
        "WER": float(summary["wer"]),
        "CER": float(summary["cer"]),
        "SentenceAcc": float(summary["sentence_accuracy"]),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics.keys(), metrics.values(), color=["#c0392b", "#d35400", "#27ae60"])
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.set_title(f"Evaluation Metrics - {summary['model']}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out_path = out_dir / "metrics_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_wer_histogram(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(df):
        ax.hist(df["wer"], bins=min(20, max(5, len(df) // 3)), color="#2c3e50", alpha=0.85)
        ax.set_xlabel("WER per sample")
        ax.set_ylabel("Count")
        ax.set_title("WER Distribution")
    else:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()

    out_path = out_dir / "wer_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_eval_artifacts(
    df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    prefix: str | None = None,
) -> dict[str, object]:
    out_dir = ensure_dir(output_dir)
    per_file_csv = Path(out_dir) / _prefixed_name(prefix, "predictions_per_file.csv")
    summary_csv = Path(out_dir) / _prefixed_name(prefix, "metrics_global.csv")

    df.to_csv(per_file_csv, index=False)
    summary = build_eval_summary(df, model_name=model_name)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    metrics_png = Path(out_dir) / _prefixed_name(prefix, "metrics_bar.png")
    wer_png = Path(out_dir) / _prefixed_name(prefix, "wer_distribution.png")

    # Generate plots using temporary default names then move to prefixed target if needed.
    raw_metrics_png = _save_metrics_bar(summary, Path(out_dir))
    raw_wer_png = _save_wer_histogram(df, Path(out_dir))
    if raw_metrics_png != metrics_png:
        raw_metrics_png.replace(metrics_png)
    if raw_wer_png != wer_png:
        raw_wer_png.replace(wer_png)

    return {
        "per_file_csv": str(per_file_csv),
        "summary_csv": str(summary_csv),
        "metrics_png": str(metrics_png),
        "wer_distribution_png": str(wer_png),
        "summary": summary,
    }


def write_model_comparison(
    output_dir: str,
    primary_summary: dict[str, float | int | str],
    baseline_summary: dict[str, float | int | str],
) -> str:
    out_dir = ensure_dir(output_dir)
    comparison_csv = Path(out_dir) / "comparison_metrics.csv"

    primary_wer = float(primary_summary["wer"])
    primary_cer = float(primary_summary["cer"])
    primary_acc = float(primary_summary["sentence_accuracy"])
    baseline_wer = float(baseline_summary["wer"])
    baseline_cer = float(baseline_summary["cer"])
    baseline_acc = float(baseline_summary["sentence_accuracy"])

    rows = [
        {
            "kind": "primary",
            "model": str(primary_summary["model"]),
            "rows": int(primary_summary["rows"]),
            "wer": primary_wer,
            "cer": primary_cer,
            "sentence_accuracy": primary_acc,
            "wer_delta_vs_baseline": primary_wer - baseline_wer,
            "cer_delta_vs_baseline": primary_cer - baseline_cer,
            "sentence_accuracy_delta_vs_baseline": primary_acc - baseline_acc,
        },
        {
            "kind": "baseline",
            "model": str(baseline_summary["model"]),
            "rows": int(baseline_summary["rows"]),
            "wer": baseline_wer,
            "cer": baseline_cer,
            "sentence_accuracy": baseline_acc,
            "wer_delta_vs_baseline": 0.0,
            "cer_delta_vs_baseline": 0.0,
            "sentence_accuracy_delta_vs_baseline": 0.0,
        },
    ]
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)
    return str(comparison_csv)
