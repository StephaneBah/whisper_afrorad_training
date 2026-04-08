from __future__ import annotations

import subprocess

import hydra
from omegaconf import DictConfig


def _run_step(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")


@hydra.main(version_base=None, config_path="../../configs", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    if cfg.pipeline.run_train:
        _run_step(["accelerate", "launch", "-m", "afrorad_pipeline.train", *cfg.pipeline.train_overrides])

    if cfg.pipeline.run_eval:
        _run_step(["accelerate", "launch", "-m", "afrorad_pipeline.eval", *cfg.pipeline.eval_overrides])


if __name__ == "__main__":
    main()
