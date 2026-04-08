from __future__ import annotations

import logging
from pathlib import Path

import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .data_pipeline import (
    build_preprocess_fn,
    cast_audio_column,
    load_afrorad_dataset,
    maybe_filter_with_manifest,
)
from .io_utils import dump_json, ensure_dir
from .layer_policy import apply_encoder_freeze_policy, resolve_unfreeze_layers
from .security import get_env_token
from .training_utils import build_differential_optimizer

LOGGER = logging.getLogger(__name__)


def _build_compute_metrics(processor):
    metric = evaluate.load("wer")

    def _compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100.0 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return _compute_metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Train config:\n%s", OmegaConf.to_yaml(cfg))

    hf_token = get_env_token(cfg.hf.token_env, required=cfg.data.use_auth_token)

    dataset = load_afrorad_dataset(cfg, token=hf_token)
    train_ds = dataset[cfg.data.train_split]
    eval_ds = dataset[cfg.data.test_split]

    if cfg.eval.use_manifest_for_eval:
        eval_ds = maybe_filter_with_manifest(
            eval_ds,
            audio_column=cfg.data.audio_column,
            manifest_path=cfg.eval.manifest_path,
            fail_on_missing=cfg.eval.fail_on_manifest_missing,
        )

    if cfg.training.max_train_samples is not None:
        train_ds = train_ds.select(range(min(cfg.training.max_train_samples, len(train_ds))))
    if cfg.training.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(cfg.training.max_eval_samples, len(eval_ds))))

    train_ds = cast_audio_column(train_ds, cfg.data.audio_column, cfg.data.sample_rate)
    eval_ds = cast_audio_column(eval_ds, cfg.data.audio_column, cfg.data.sample_rate)

    processor = WhisperProcessor.from_pretrained(
        cfg.model.processor_name,
        language=cfg.model.language,
        task=cfg.model.task,
        token=hf_token,
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.name, token=hf_token)

    encoder_layers = len(model.model.encoder.layers)
    unfreeze_layers = resolve_unfreeze_layers(
        model_name=cfg.model.name,
        encoder_layer_count=encoder_layers,
        requested=cfg.training.encoder_unfreeze_layers,
    )
    policy_stats = apply_encoder_freeze_policy(model, unfreeze_layers)
    LOGGER.info(
        "Using encoder_unfreeze_layers=%s (total encoder layers=%s, trainable=%s/%s)",
        unfreeze_layers,
        encoder_layers,
        policy_stats["trainable"],
        policy_stats["total"],
    )

    preprocess_fn = build_preprocess_fn(cfg, processor)
    remove_cols_train = train_ds.column_names
    remove_cols_eval = eval_ds.column_names

    train_prepared = train_ds.map(
        preprocess_fn,
        remove_columns=remove_cols_train,
        num_proc=cfg.training.num_proc,
    )
    eval_prepared = eval_ds.map(
        preprocess_fn,
        remove_columns=remove_cols_eval,
        num_proc=cfg.training.num_proc,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    output_dir = str(ensure_dir(cfg.training.output_dir))
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=cfg.training.seed,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        evaluation_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        predict_with_generate=cfg.training.predict_with_generate,
        generation_max_length=cfg.training.generation_max_length,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=cfg.logging.report_to,
        push_to_hub=cfg.hf.push_to_hub,
        hub_model_id=cfg.hf.hub_model_id if cfg.hf.push_to_hub else None,
    )

    optimizer = build_differential_optimizer(model, cfg)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_prepared,
        eval_dataset=eval_prepared,
        data_collator=data_collator,
        compute_metrics=_build_compute_metrics(processor),
        tokenizer=processor.tokenizer,
        optimizers=(optimizer, None),
    )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    dump_json(
        {
            "eval_metrics": eval_metrics,
            "encoder_unfreeze_layers": unfreeze_layers,
            "model_name": cfg.model.name,
        },
        str(Path(output_dir) / "train_metrics.json"),
    )

    if cfg.hf.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
