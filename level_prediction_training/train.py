import json
import logging
import logging.config
import os
import pickle
import random
from pathlib import Path

import hydra
import polars as pl
import torch
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from src.config import Config
from src.dataset import TimeSeriesDataset, load_training_data
from src.quantile_loss import quantile_loss
from src.ts_mixer import TSMixer

os.environ["WANDB_SILENT"] = "true"


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


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(config: Config):
    setup_logging()
    log = logging.getLogger("training")

    seed = config.seed if config.seed is not None else random.randint(0, 2**16 - 1)
    log.info(f"Using seed: {seed}")
    torch.manual_seed(seed)

    stations = pl.read_json(config.stations_filepath)

    config.model_save_dir = Path(config.model_save_dir)

    wandb.init(
        project="river-level-forecasting",
        config={
            **OmegaConf.to_container(config),
            "stations": stations.to_dict(),
            "seed": seed,
        },
    )

    log.info(f"wandb run: {wandb.run.get_url()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    log.info("Loading training data")

    train_df, test_df = load_training_data(stations, train_split=config.train_split)

    log.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")

    log.info("Preprocessing data")
    X_preprocessing = make_column_transformer(
        (
            StandardScaler(),
            make_column_selector(pattern=r"- level \(m\)"),
        ),
        (
            MinMaxScaler(),
            make_column_selector(pattern=r"- rainfall \(mm\)"),
        ),
        remainder="passthrough",
    )

    y_preprocessing = StandardScaler()

    X_train = train_df.to_pandas().astype("float32")
    y_train = train_df.select(config.target_col).to_numpy().astype("float32")

    X_test = test_df.to_pandas().astype("float32")
    y_test = test_df.select(config.target_col).to_numpy().astype("float32")

    if config.predict_difference:
        y_train = y_train[1:] - y_train[:-1]
        y_test = y_test[1:] - y_test[:-1]

        X_train = X_train[1:]
        X_test = X_test[1:]

    X_train = torch.tensor(X_preprocessing.fit_transform(X_train), device=device)
    y_train = torch.tensor(y_preprocessing.fit_transform(y_train), device=device)

    X_test = torch.tensor(X_preprocessing.transform(X_test), device=device)
    y_test = torch.tensor(y_preprocessing.transform(y_test), device=device)

    train_dataset = TimeSeriesDataset(
        X_train, y_train, config.context_length, config.prediction_length
    )
    test_dataset = TimeSeriesDataset(
        X_test, y_test, config.context_length, config.prediction_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    steps_per_train_epoch = len(train_loader)
    steps_per_test_epoch = len(test_loader)

    log.info("Initialising model")

    model = TSMixer(
        config.context_length,
        config.prediction_length,
        input_channels=X_train.shape[1],
        output_channels=len(config.quantiles) + 1,
        activation_fn=config.activation_function,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        ff_dim=config.ff_dim,
        normalize_before=config.normalize_before,
        norm_type=config.norm_type,
        predict_difference=config.predict_difference,
    ).to(device)

    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap >= (7, 0):
            model = torch.compile(model)

    log.info(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    optim = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    quantiles = torch.tensor(config.quantiles, device=device)

    metric_names = [
        "mse loss",
        "quantile loss",
        "total loss",
    ]

    train_metrics = torch.zeros(len(metric_names), device=device)
    val_metrics = torch.zeros(len(metric_names), device=device)

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    training_task = progress.add_task(
        "[cyan]Training",
        total=config.train_epochs * (steps_per_train_epoch + steps_per_test_epoch),
    )

    def log_metrics():
        metrics_dict = {
            f"{metric_name} ({type})": metric.item()
            for metrics, type, len in [
                (train_metrics, "train", len(train_loader)),
                (val_metrics, "val", len(test_loader)),
            ]
            for metric, metric_name in zip(metrics.cpu() / len, metric_names)
        }

        wandb.log(metrics_dict)

        log.info({key: f"{value:.4f}" for key, value in metrics_dict.items()})

    def loss_func(y_pred, y_true):
        pred_mean, pred_quantiles = y_pred[..., 0:1], y_pred[..., 1:]

        mean_loss = F.mse_loss(pred_mean, y_true)
        quantiles_loss = quantile_loss(
            y_true,
            pred_quantiles,
            quantiles,
        )

        total_loss = mean_loss + quantiles_loss

        return total_loss, torch.stack([mean_loss, quantiles_loss, total_loss]).detach()

    log.info("Starting training")

    with progress:
        for epoch in range(config.train_epochs):
            model.train()

            log.info(f"Epoch {epoch + 1} / {config.train_epochs}")

            epoch_training_task = progress.add_task(
                f"[green]Epoch {epoch + 1} (train)", total=steps_per_train_epoch
            )

            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)

                loss, metrics = loss_func(y_pred, y_batch)

                loss.backward()
                train_metrics += metrics

                optim.step()
                optim.zero_grad()

                progress.update(epoch_training_task, advance=1)
                progress.update(training_task, advance=1)

            model.eval()

            progress.remove_task(epoch_training_task)
            epoch_test_task = progress.add_task(
                f"[green]Epoch {epoch + 1} (test)", total=steps_per_test_epoch
            )

            for X_val_batch, y_val_batch in test_loader:
                with torch.no_grad():
                    y_val_pred = model(X_val_batch)
                    _, metrics = loss_func(y_val_pred, y_val_batch)

                val_metrics += metrics

                progress.update(epoch_test_task, advance=1)
                progress.update(training_task, advance=1)

            progress.remove_task(epoch_test_task)

            log_metrics()
            train_metrics.zero_()
            val_metrics.zero_()

    log.info("Training complete")

    model_dir: Path = config.model_save_dir / wandb.run.name

    if not model_dir.exists():
        os.makedirs(model_dir)

    model_file_path = model_dir / "model_torchscript.pt"

    log.info(f"Saving model to {model_file_path}")

    model.eval()
    model_torchscript = torch.jit.script(model.to("cpu"))
    model_torchscript = torch.jit.optimize_for_inference(model_torchscript)
    model_torchscript.save(model_file_path)

    wandb.log_artifact(model_file_path, name="trained_model", type="model")

    preprocessing = {
        "X": X_preprocessing,
        "y": y_preprocessing,
    }

    preprocessing_file_path = model_dir / "preprocessing.pickle"

    log.info(f"Saving preprocessing pipeline to {preprocessing_file_path}")

    with open(preprocessing_file_path, "wb") as f:
        pickle.dump(preprocessing, f)

    wandb.log_artifact(
        preprocessing_file_path, name="preprocessing_pipeline", type="model"
    )

    inference_config_file_path = model_dir / "inference_config.json"

    inference_config = dict(
        prediction_length=config.prediction_length,
        context_length=config.context_length,
        quantiles=list(config.quantiles),
        target_col=config.target_col,
        predict_difference=config.predict_difference,
        stations=stations.to_dict(as_series=False),
    )

    with open(inference_config_file_path, "w") as f:
        json.dump(inference_config, f)

    wandb.log_artifact(
        inference_config_file_path, name="inference_config", type="config"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
