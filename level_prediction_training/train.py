import logging
import logging.config
import os
import random
from pathlib import Path

import polars as pl
import torch
import yaml


import wandb
from src.config import Config
from src.dataset import DataModule
import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer, loggers, disable_possible_user_warnings
import warnings

disable_possible_user_warnings()
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore", message="There is a wandb run already in progress")

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


def get_model(config, input_channels):
    from src.models.mlp import TimeSeriesMLP
    from src.models.ts_mixer import TSMixer
    
    model_classes = {
        "mlp": TimeSeriesMLP,
        "ts-mixer": TSMixer,
    }
    
    try:
        model_class = model_classes[config.model.name]
    except KeyError:
        raise ValueError(f"Unknown model name: {config.model.name}")
        
    return model_class(
        config=config,
        input_channels=input_channels,
    )


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(config: Config):
    
    setup_logging()
    log = logging.getLogger("training")

    seed = config.seed if config.seed is not None else random.randint(0, 2**16 - 1)
    log.info(f"Using seed: {seed}")
    torch.manual_seed(seed)

    config.model_save_dir = Path(config.model_save_dir)

    wandb.init(
        project="river-level-forecasting",
        config={
            **OmegaConf.to_container(config),
            "seed": seed,
        },
    )

    log.info(f"wandb run: {wandb.run.get_url()}")

    
    data_module = DataModule(config)

    model = get_model(config, data_module.num_features)
    
    trainer = Trainer(
        max_epochs=config.train_epochs,
        logger=loggers.WandbLogger(),
        benchmark=True,
    )
    
    trainer.fit(model, data_module)

    # model_dir: Path = config.model_save_dir / wandb.run.name

    # if not model_dir.exists():
    #     os.makedirs(model_dir)

    # model_file_path = model_dir / "model_torchscript.pt"

    # log.info(f"Saving model to {model_file_path}")

    # for model in models:
    #     model.eval()

    # model_torchscript = torch.jit.script(model.to("cpu"))
    # model_torchscript = torch.jit.optimize_for_inference(model_torchscript)
    # model_torchscript.save(model_file_path)

    # wandb.log_artifact(model_file_path, name="trained_model", type="model")

    # preprocessing = {
    #     "X": X_preprocessing,
    #     "y": y_preprocessing,
    # }

    # preprocessing_file_path = model_dir / "preprocessing.pickle"

    # log.info(f"Saving preprocessing pipeline to {preprocessing_file_path}")

    # with open(preprocessing_file_path, "wb") as f:
    #     pickle.dump(preprocessing, f)

    # wandb.log_artifact(
    #     preprocessing_file_path, name="preprocessing_pipeline", type="model"
    # )

    # inference_config = dict(
    #     prediction_length=config.prediction_length,
    #     context_length=config.context_length,
    #     target_col=config.target_col,
    #     predict_difference=config.predict_difference,
    #     stations=stations.to_dict(as_series=False),
    # )

    # inference_config = dict(
    #     prediction_length=config.prediction_length,
    #     context_length=config.context_length,
    #     quantiles=list(config.quantiles),
    #     target_col=config.target_col,
    #     predict_difference=config.predict_difference,
    #     stations=stations.to_dict(as_series=False),
    # )

    # with open(inference_config_file_path, "w") as f:
    #     json.dump(inference_config, f)

    # wandb.log_artifact(
    #     inference_config_file_path, name="inference_config", type="config"
    # )

    wandb.finish()


if __name__ == "__main__":
    main()
