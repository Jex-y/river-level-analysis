import logging
import os
import pickle
import random
from pathlib import Path

import hydra
import polars as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import Config
from src.dataset import TimeSeriesDataset, load_training_data
from src.quantile_loss import quantile_loss
from src.ts_mixer import TSMixer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['WANDB_SILENT'] = 'true'


httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)


@hydra.main(version_base=None, config_name='config', config_path='.')
def main(config: Config):
    seed = config.seed if config.seed is not None else random.randint(0, 1000)
    log.info(f'Using seed: {seed}')
    torch.manual_seed(seed)

    stations = pl.read_csv(config.stations_filepath)

    config.model_save_dir = Path(config.model_save_dir)

    wandb.init(
        project='river-level-forecasting',
        config={
            **OmegaConf.to_container(config),
            'stations': stations.to_dict(),
            'seed': seed,
        },
    )

    log.info(f'wandb run: {wandb.run.get_url()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    train_df, test_df = load_training_data(stations, train_split=config.train_split)

    X_preprocessing = make_column_transformer(
        (
            StandardScaler(),
            make_column_selector(pattern=r'- level \(m\)'),
        ),
        (
            MinMaxScaler(),
            make_column_selector(pattern=r'- rainfall \(mm\)'),
        ),
        remainder='passthrough',
    )

    y_preprocessing = StandardScaler()

    X_train = train_df.to_pandas().astype('float32')
    y_train = train_df.select(config.target_col).to_numpy().astype('float32')

    X_test = test_df.to_pandas().astype('float32')
    y_test = test_df.select(config.target_col).to_numpy().astype('float32')

    X_train = torch.tensor(X_preprocessing.fit_transform(X_train)).to(device)
    y_train = torch.tensor(y_preprocessing.fit_transform(y_train)).to(device)

    X_test = torch.tensor(X_preprocessing.transform(X_test)).to(device)
    y_test = torch.tensor(y_preprocessing.transform(y_test)).to(device)

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
        # persistent_workers=True,
        # num_workers=1,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        # persistent_workers=True,
        # num_workers=1,
    )

    steps_per_epoch = len(train_loader) + len(test_loader)

    model = TSMixer(
        config.context_length,
        config.prediction_length,
        input_channels=X_train.shape[1],
        output_channels=len(config.quantiles),
        activation_fn=config.activation_function,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        ff_dim=config.ff_dim,
        normalize_before=config.normalize_before,
        norm_type=config.norm_type,
    ).to(device)

    optim = AdamW(model.parameters(), lr=config.lr)

    quantiles = torch.tensor(config.quantiles, device=device)
    train_metrics = torch.zeros(len(quantiles) + 2, device=device)
    val_metrics = torch.zeros(len(quantiles) + 2, device=device)

    metric_names = [f'loss_quantile_{q:.2f}' for q in config.quantiles] + [
        'loss_total',
        'mae',
    ]

    overall_pbar = tqdm(
        total=config.train_epochs * steps_per_epoch * 2,
        desc='Training',
        unit='step',
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {rate_fmt}{postfix} [{elapsed}<{remaining}]',
        smoothing=0.01,
    )

    def log_metrics():
        metrics_dict = {
            f'{type}_{metric_name}': metric
            for metrics, type, len in [
                (train_metrics, 'train', len(train_loader)),
                (val_metrics, 'val', len(test_loader)),
            ]
            for metric, metric_name in zip(metrics.cpu().detach() / len, metric_names)
        }

        wandb.log(metrics_dict)

        overall_pbar.set_postfix(
            {k: f'{v:.4f}' for k, v in metrics_dict.items() if 'loss_total' in k}
        )

    for epoch in range(config.train_epochs):
        model.train()

        for X_batch, y_batch in tqdm(
            train_loader,
            desc=f'Train Epoch {epoch + 1}',
            leave=False,
            unit='step',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {rate_fmt}{postfix} [{elapsed}<{remaining}]',
        ):
            y_pred = model(X_batch)

            loss = quantile_loss(y_batch, y_pred, quantiles)
            mean_loss = loss.mean()
            mean_loss.backward()
            optim.step()
            optim.zero_grad()

            train_metrics[: len(quantiles)] += loss.detach()
            train_metrics[len(quantiles) + 0] += mean_loss.detach()
            train_metrics[len(quantiles) + 1] += (
                torch.abs(y_batch - y_pred[..., 1:2]).detach().mean()
            )

            overall_pbar.update()

        model.eval()

        for X_val_batch, y_val_batch in tqdm(
            test_loader,
            desc=f'Test Epoch {epoch + 1}',
            leave=False,
            unit='step',
        ):
            y_val_pred = model(X_val_batch)

            val_loss = quantile_loss(y_val_batch, y_val_pred, quantiles)

            val_metrics[: len(quantiles)] += val_loss.detach()
            val_metrics[len(quantiles) + 0] += val_loss.mean().detach()
            val_metrics[len(quantiles) + 1] += (
                torch.abs(y_batch - y_pred[..., 1:2]).detach().mean()
            )

            overall_pbar.update()

        log_metrics()
        train_metrics.zero_()
        val_metrics.zero_()

    overall_pbar.close()

    if not config.model_save_dir.exists():
        os.mkdir(config.model_save_dir)

    model_file_path = config.model_save_dir / f'{wandb.run.name}_torchscript.pt'

    model_torchscript = torch.jit.script(model)
    model_torchscript.save(model_file_path)

    wandb.log_artifact(model_file_path, name='trained_model', type='model')

    preprocessing = {
        'X': X_preprocessing,
        'y': y_preprocessing,
    }

    preprocessing_file_path = (
        config.model_save_dir / f'{wandb.run.name}_preprocessing.pickle'
    )

    with open(preprocessing_file_path, 'wb') as f:
        pickle.dump(preprocessing, f)

    wandb.log_artifact(
        preprocessing_file_path, name='preprocessing_pipeline', type='model'
    )

    wandb.finish()


if __name__ == '__main__':
    main()
