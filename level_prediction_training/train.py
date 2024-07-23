import logging
import logging.config
import os
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
from torch.utils.data import DataLoader, default_collate

import wandb
from src.config import Config
from src.dataset import TimeSeriesDataset, load_training_data
from src.ts_mixer import Ensemble, TSMixer

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

    model = Ensemble(
        lambda: TSMixer(
            config.context_length,
            config.prediction_length,
            input_channels=X_train.shape[1],
            activation_fn=config.activation_function,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            ff_dim=config.ff_dim,
            normalize_before=config.normalize_before,
            norm_type=config.norm_type,
            predict_difference=config.predict_difference,
        ).to(device),
        config.ensemble_size,
    )

    optim = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
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

    log.info("Starting training")

    def loss_func(y_true, y_pred):
        per_model_loss = F.smooth_l1_loss(
            y_pred,
            y_true.repeat(config.ensemble_size, *([1] * y_true.ndim)),
            reduction="none",
        ).mean(dim=tuple(range(1, y_pred.ndim)))

        y_pred_mean = y_pred.mean(dim=0)
        y_pred_std = y_pred.std(dim=0)

        y_pred_mean_error = F.l1_loss(y_pred_mean, y_true, reduction="none").mean(
            dim=tuple(range(1, y_true.ndim))
        )
        error_in_1std_freq = (y_pred_mean_error > y_pred_std).float().mean()

        return (
            per_model_loss.mean(),
            {
                "Mean loss": per_model_loss.mean(),
                "MAE from ensemble mean": y_pred_mean_error.mean(),
                "error in 1std freq": error_in_1std_freq,
            }
            | {f"model {i+1} loss": loss for i, loss in enumerate(per_model_loss)},
        )

    def mean_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        collated_dicts: dict[str, torch.Tensor] = default_collate(list(dicts))
        return {key: val.mean().item() for key, val in collated_dicts.items()}

    with progress:
        for epoch in range(config.train_epochs):
            model.train()

            log.info(f"Epoch {epoch + 1} / {config.train_epochs}")

            epoch_training_task = progress.add_task(
                f"[green]Epoch {epoch + 1} (train)", total=steps_per_train_epoch
            )

            train_metric_log = []

            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)

                loss, metrics = loss_func(y_batch, y_pred)
                train_metric_log.append(metrics)

                loss.mean().backward()

                optim.step()
                optim.zero_grad()

                progress.update(epoch_training_task, advance=1)
                progress.update(training_task, advance=1)

            model.eval()
            val_metric_log = []

            progress.remove_task(epoch_training_task)
            epoch_test_task = progress.add_task(
                f"[green]Epoch {epoch + 1} (test)", total=steps_per_test_epoch
            )

            for X_val_batch, y_val_batch in test_loader:
                with torch.no_grad():
                    y_val_pred = model(X_val_batch)

                    _, val_metrics = loss_func(y_val_batch, y_val_pred)

                    val_metric_log.append(val_metrics)

                progress.update(epoch_test_task, advance=1)
                progress.update(training_task, advance=1)

            progress.remove_task(epoch_test_task)
            mean_train_metrics = mean_dicts(train_metric_log)
            mean_val_metrics = mean_dicts(val_metric_log)

            wandb.log(
                {
                    "train": mean_train_metrics,
                    "val": mean_val_metrics,
                }
            )

            mean_train_metrics = {
                key: f"{val:.4f}" for key, val in mean_train_metrics.items()
            }
            mean_val_metrics = {
                key: f"{val:.4f}" for key, val in mean_val_metrics.items()
            }

            log.info(f"Train metrics: {mean_train_metrics}")
            log.info(f"Validation metrics: {mean_val_metrics}")

    log.info("Training complete")

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

    # inference_config_file_path = model_dir / "inference_config.json"

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
