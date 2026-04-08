from __future__ import annotations

import logging

import evaluate
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .data_pipeline import cast_audio_column, load_afrorad_dataset, maybe_filter_with_manifest, normalize_text
from .reporting import write_eval_artifacts, write_model_comparison
from .security import get_env_token

LOGGER = logging.getLogger(__name__)


def _transcribe_one(model, processor, audio_array, sample_rate: int, device: str, max_new_tokens: int) -> str:
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    predicted_ids = model.generate(input_features, max_new_tokens=max_new_tokens)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def _evaluate_model(cfg: DictConfig, eval_ds, hf_token: str | None, model_name: str) -> pd.DataFrame:
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=cfg.model.language,
        task=cfg.model.task,
        token=hf_token,
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name, token=hf_token)

    device = cfg.runtime.device
    model.to(device)
    model.eval()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    rows = []
    for sample in eval_ds:
        audio = sample[cfg.data.audio_column]
        reference = normalize_text(
            sample[cfg.data.text_column],
            lower=cfg.preprocessing.lowercase,
            strip_newlines=cfg.preprocessing.strip_newlines,
        )
        prediction = _transcribe_one(
            model=model,
            processor=processor,
            audio_array=audio["array"],
            sample_rate=audio["sampling_rate"],
            device=device,
            max_new_tokens=cfg.eval.max_new_tokens,
        )
        prediction = normalize_text(
            prediction,
            lower=cfg.preprocessing.lowercase,
            strip_newlines=cfg.preprocessing.strip_newlines,
        )

        row_wer = 100.0 * wer_metric.compute(predictions=[prediction], references=[reference])
        row_cer = 100.0 * cer_metric.compute(predictions=[prediction], references=[reference])
        sent_acc = 100.0 if prediction == reference else 0.0

        rows.append(
            {
                "reference": reference,
                "prediction": prediction,
                "wer": row_wer,
                "cer": row_cer,
                "sentence_accuracy": sent_acc,
            }
        )

    return pd.DataFrame(rows)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Eval config:\n%s", OmegaConf.to_yaml(cfg))

    hf_token = get_env_token(cfg.hf.token_env, required=cfg.data.use_auth_token)

    dataset = load_afrorad_dataset(cfg, token=hf_token)
    eval_ds = dataset[cfg.data.test_split]

    if cfg.eval.use_manifest_for_eval:
        eval_ds = maybe_filter_with_manifest(
            eval_ds,
            audio_column=cfg.data.audio_column,
            manifest_path=cfg.eval.manifest_path,
            fail_on_missing=cfg.eval.fail_on_manifest_missing,
        )

    if cfg.eval.max_samples is not None:
        eval_ds = eval_ds.select(range(min(cfg.eval.max_samples, len(eval_ds))))

    eval_ds = cast_audio_column(eval_ds, cfg.data.audio_column, cfg.data.sample_rate)

    model_name = cfg.eval.model_path_or_name
    primary_df = _evaluate_model(cfg=cfg, eval_ds=eval_ds, hf_token=hf_token, model_name=model_name)
    artifacts = write_eval_artifacts(df=primary_df, output_dir=cfg.eval.output_dir, model_name=model_name)

    LOGGER.info("Saved per-file results to %s", artifacts["per_file_csv"])
    LOGGER.info("Saved global metrics to %s", artifacts["summary_csv"])
    LOGGER.info("Saved metrics figure to %s", artifacts["metrics_png"])
    LOGGER.info("Saved WER distribution figure to %s", artifacts["wer_distribution_png"])

    if cfg.eval.compare_with_baseline:
        baseline_model = cfg.eval.baseline_model_path_or_name
        if not baseline_model:
            raise ValueError("eval.baseline_model_path_or_name must be set when eval.compare_with_baseline=true")

        LOGGER.info("Running baseline comparison with model=%s", baseline_model)
        baseline_df = _evaluate_model(
            cfg=cfg,
            eval_ds=eval_ds,
            hf_token=hf_token,
            model_name=baseline_model,
        )
        baseline_artifacts = write_eval_artifacts(
            df=baseline_df,
            output_dir=cfg.eval.output_dir,
            model_name=baseline_model,
            prefix="baseline",
        )
        comparison_csv = write_model_comparison(
            output_dir=cfg.eval.output_dir,
            primary_summary=artifacts["summary"],
            baseline_summary=baseline_artifacts["summary"],
        )
        LOGGER.info("Saved baseline per-file results to %s", baseline_artifacts["per_file_csv"])
        LOGGER.info("Saved baseline metrics to %s", baseline_artifacts["summary_csv"])
        LOGGER.info("Saved baseline comparison to %s", comparison_csv)


if __name__ == "__main__":
    main()
