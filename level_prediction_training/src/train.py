import logging
import logging.config
import os
import random
import warnings
from dataclasses import asdict
from typing import Optional

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
from polars import DataFrame
from pathlib import Path
from torch.nn import Module as TorchModule


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
    model: TorchModule,
    stations: DataFrame,
    config: Config,
):
    import json

    log = logging.getLogger("training")

    model_dir: Path = config.model_save_dir / wandb.run.name

    if not model_dir.exists():
        os.makedirs(model_dir)

    model_file_path = model_dir / "model_torchscript.pt"
    inference_config_file_path = model_dir / "inference_config.json"

    # Save model as TorchScript

    log.info(f"Saving model to {model_file_path}")

    model_torchscript = torch.jit.script(model.to("cpu").eval())
    model_torchscript = torch.jit.optimize_for_inference(model_torchscript)
    model_torchscript.save(model_file_path)

    # Save config for inference

    inference_config = dict(
        target_col=config.target_col,
        prediction_length=config.prediction_length,
        context_length=config.context_length,
        quantiles=list(config.quantiles),
        thresholds=list(config.thresholds),
        stations=stations.to_dict(as_series=False),
    )

    with open(inference_config_file_path, "w") as f:
        json.dump(inference_config, f)

    # Add artifacts to wandb

    wandb.log_artifact(model_file_path, name="trained_model", type="model")
    wandb.log_artifact(
        inference_config_file_path, name="inference_config", type="config"
    )


def train(config: Optional[Config] = None):
    setup_logging()
    log = logging.getLogger("training")

    if config is None:
        seed = random.randint(0, 2**16 - 1)
        wandb.init(project="river-level-forecasting", config={"seed": seed})

        wandb_config = wandb.config
        # Sometimes rolling windows are not passed as an iterable
        wandb_config["rolling_windows"] = (
            (wandb_config["rolling_windows"],)
            if isinstance(wandb_config["rolling_windows"], int)
            else tuple(wandb_config["rolling_windows"])
        )

        config = Config(**wandb_config)
    else:
        wandb.init(project="river-level-forecasting", config=asdict(config))
        seed = config.seed if config.seed is not None else random.randint(0, 2**16 - 1)

    log.info(f"Using config: {config}")

    seed_everything(seed)

    log.info(f"wandb run: {wandb.run.get_url()}")

    data_module = DataModule(config)

    model = TimeSeriesModel(
        input_column_names=data_module.x_column_names, config=config
    ).fit_preprocessing(data_module.train_dataset.x, data_module.train_dataset.y)

    trainer = Trainer(
        max_epochs=config.train_epochs,
        logger=loggers.WandbLogger(),
        benchmark=True,
        fast_dev_run=config.dev_run,
    )

    trainer.fit(model, data_module)

    # save_model(model, data_module.stations, config)

    wandb.finish()
