# AfroRad Whisper Pipeline

Training and evaluation pipeline for Whisper models on StephaneBah/AfroRadVoice-FR.
The workflow is configuration-driven (Hydra) and can run through Makefile commands or a PowerShell wrapper.

## Project Context

### Dataset

- Source: https://huggingface.co/datasets/StephaneBah/AfroRadVoice-FR
- Access type: gated access (conditions must be accepted on Hugging Face)
- Task and language: automatic speech recognition (ASR), French
- Domain: radiology dictation in Afro-French context
- Composition (dataset card): 562 audio samples, about 4.54 hours, with real, synthetic, and augmented recordings
- Typical split (dataset card): 487 train/validation, 75 test

### Model and adaptation approach

- Reference model card: https://huggingface.co/StephaneBah/Whisper-AfroRad-FR
- Base architecture: openai/whisper-small
- Adaptation goal:
   - acoustic robustness to African-accented French
   - improved medical/radiology term recognition
- Reported strategy: LoRA adapters targeting early encoder layers (acoustics) and decoder capacity (terminology/language structure)

### Scope of this repository

This repository provides a reproducible training/evaluation pipeline around this dataset and modeling direction.
It is designed for one-command execution and configurable experiments across runtime profiles.

## Requirements

- Access granted to StephaneBah/AfroRadVoice-FR on Hugging Face.
- HF token available in environment variable HF_TOKEN.
- Python environment prepared with project dependencies.

## Quick Start

### Linux/macOS (with make)

```bash
make setup
export HF_TOKEN="hf_..."
make run
```

### Windows (without make)

```powershell
pwsh -File scripts/pipeline.ps1 -Task setup
$env:HF_TOKEN="hf_..."
pwsh -File scripts/pipeline.ps1 -Task run
```

## Command Reference

### Make targets

| Command | Role |
|---|---|
| make setup | Installs project and development dependencies into the selected Python environment. |
| make doctor | Validates runtime prerequisites: Python, imports, HF_TOKEN, CUDA availability. |
| make train | Runs Whisper fine-tuning with Hydra config `configs/train.yaml`. |
| make eval | Runs model evaluation with Hydra config `configs/eval.yaml`. |
| make run | Executes `doctor`, then `train`, then `eval` in sequence. |
| make test | Runs unit tests. |
| make lint | Runs Ruff lint checks on source and tests. |

### PowerShell wrapper tasks

Use the same roles as above via:

```powershell
pwsh -File scripts/pipeline.ps1 -Task setup
pwsh -File scripts/pipeline.ps1 -Task doctor
pwsh -File scripts/pipeline.ps1 -Task train
pwsh -File scripts/pipeline.ps1 -Task eval
pwsh -File scripts/pipeline.ps1 -Task run
pwsh -File scripts/pipeline.ps1 -Task test
pwsh -File scripts/pipeline.ps1 -Task lint
```

## Common Usage Scenarios

### Select Whisper size

```bash
accelerate launch -m afrorad_pipeline.train model=whisper_base
accelerate launch -m afrorad_pipeline.train model=whisper_small
accelerate launch -m afrorad_pipeline.train model=whisper_medium
```

### Override number of trainable encoder layers

```bash
accelerate launch -m afrorad_pipeline.train model=whisper_medium training.encoder_unfreeze_layers=8
```

### Use Windows GPU runtime profile

```bash
accelerate launch -m afrorad_pipeline.train runtime=windows_gpu
```

### Evaluate a fine-tuned model

```bash
accelerate launch -m afrorad_pipeline.eval eval.model_path_or_name=StephaneBah/whisper-small-rad-FR2
```

### Compare fine-tuned model versus baseline

```bash
accelerate launch -m afrorad_pipeline.eval \
   eval.model_path_or_name=StephaneBah/whisper-small-rad-FR2 \
   eval.compare_with_baseline=true \
   eval.baseline_model_path_or_name=openai/whisper-small
```

### Evaluate only a manifest-defined subset

```bash
accelerate launch -m afrorad_pipeline.eval \
   eval.use_manifest_for_eval=true \
   eval.manifest_path=manifests/test_manifest.example.json \
   eval.fail_on_manifest_missing=true
```

## Evaluation Outputs

Default output directory: `outputs/eval`

- predictions_per_file.csv
- metrics_global.csv
- metrics_bar.png
- wer_distribution.png

When baseline comparison is enabled:

- baseline_predictions_per_file.csv
- baseline_metrics_global.csv
- baseline_metrics_bar.png
- baseline_wer_distribution.png
- comparison_metrics.csv

## Test Manifest Format

Legacy format (supported):

```json
{
   "audios": [
      "report_001/recorded_audio.wav",
      "report_002/recorded_audio.wav"
   ]
}
```

Structured format (recommended):

```json
{
   "manifest_version": 1,
   "entries": [
      {"id": "r1", "audio": "report_001/recorded_audio.wav", "split": "test"},
      {"id": "r2", "audio": "report_002/recorded_audio.wav", "split": "test"}
   ]
}
```

Reference example: manifests/test_manifest.example.json

## Security Notes

- Never commit tokens.
- Provide HF_TOKEN via environment variable only.
- Rotate credentials immediately if any token was exposed.
