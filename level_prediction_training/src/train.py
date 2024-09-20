import logging
import logging.config
import os
import random
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import polars as pl
import torch
import yaml
from pytorch_lightning import (
    Trainer,
    disable_possible_user_warnings,
    loggers,
    seed_everything,
)

import wandb

from .config import Config
from .dataset import DataModule
from .model import TimeSeriesModel


def quiet_output():
    disable_possible_user_warnings()
    os.environ["WANDB_SILENT"] = "true"
    warnings.filterwarnings(
        "ignore", message="There is a wandb run already in progress"
    )


def setup_logging(default_path="logging.yaml", default_level=logging.INFO):
    """Setup logging configuration"""
    path = default_path
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging configuration file not found at {path}")


def save_model(
    model: TimeSeriesModel,
    stations: pl.DataFrame,
    config: Config,
):
    import json

    log = logging.getLogger("training")

    model_dir: Path = config.model_save_dir / wandb.run.name

    if not model_dir.exists():
        os.makedirs(model_dir)

    model_file_path = model_dir / "model_torchscript.onnx"
    inference_config_file_path = model_dir / "inference_config.json"

    # Save model as ONNX

    log.info(f"Saving model to {model_file_path}")

    onnx_program = torch.onnx.dynamo_export(
        model.to("cpu").eval().forecast, *model.get_example_forecast_input()
    )
    onnx_program.save(str(model_file_path))

    # Save config for inference

    inference_config = dict(
        prediction_length=config.prediction_length,
        prev_timesteps=model.required_samples,
        input_columns=stations.select(
            pl.col("flooding_api_notation").alias("station_id"),
            pl.col("parameter"),
            pl.col("label"),
        ).to_dicts(),
        thresholds=list(config.thresholds),
    )

    with open(inference_config_file_path, "w") as f:
        json.dump(inference_config, f)


def train(config: Optional[Config] = None):
    setup_logging()
    log = logging.getLogger("training")

    if config is None:
        seed = random.randint(0, 2**16 - 1)
        wandb.init(project="river-level-forecasting", config={"seed": seed})

        config_dict = dict(**wandb.config)

        config = Config(**config_dict)
    else:
        wandb.init(project="river-level-forecasting", config=asdict(config))
        seed = config.seed if config.seed is not None else random.randint(0, 2**16 - 1)

    # make sure that config.rolling_windows is a tuple
    if not isinstance(config.rolling_windows, tuple):
        config.rolling_windows = (
            tuple(config.rolling_windows)
            if isinstance(config.rolling_windows, (list, set))
            else tuple(
                config.rolling_windows,
            )
        )

    log.info(f"Using config: {config}")

    seed_everything(seed)

    log.info(f"wandb run: {wandb.run.get_url()}")

    data_module = DataModule(config)

    model = TimeSeriesModel(
        input_column_names=data_module.x_column_names, config=config
    )

    trainer = Trainer(
        max_epochs=config.train_epochs,
        logger=loggers.WandbLogger(),
        fast_dev_run=config.dev_run,
    )

    trainer.fit(model, data_module)

    save_model(model, data_module.stations, config)

    wandb.finish()
