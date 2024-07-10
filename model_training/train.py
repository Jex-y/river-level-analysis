import logging
import os
import random
from collections import defaultdict

import hydra
import polars as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import Config
from src.dataset import TimeSeriesDataset, load_data
from src.quantile_loss import quantile_loss
from src.ts_mixer import TSMixer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['WANDB_SILENT'] = 'true'


httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)


@hydra.main(version_base=None, config_name='config', config_path='.')
def main(config: Config):
    seed = config.seed if config.seed is not None else random.randint(0, 1000)
    log.info(f'Using seed: {seed}')
    torch.manual_seed(seed)

    wandb.init(
        project='river-level-forecasting',
        config={
            **OmegaConf.to_container(config),
            'seed': seed,
        },
    )

    train_df, test_df = load_data(train_split=config.train_split)

    X_preprocessing = make_column_transformer(
        (
            StandardScaler(),
            make_column_selector(pattern='level-i-900-m'),
        ),
        (
            MinMaxScaler(),
            make_column_selector(pattern='rainfall-t-900-mm'),
        ),
        remainder='passthrough',
    )

    X_train = (
        train_df.select(pl.col('*').exclude('timestamp')).to_pandas().astype('float32')
    )
    y_train = (
        train_df.select('Durham New Elvet Bridge level-i-900-m')
        .to_numpy()
        .astype('float32')
    )

    X_test = (
        test_df.select(pl.col('*').exclude('timestamp')).to_pandas().astype('float32')
    )
    y_test = (
        test_df.select('Durham New Elvet Bridge level-i-900-m')
        .to_numpy()
        .astype('float32')
    )

    y_preprocessing = StandardScaler()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

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
        persistent_workers=True,
        num_workers=1,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        persistent_workers=True,
        num_workers=1,
    )

    steps_per_epoch = len(train_loader)

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

    overall_pbar = tqdm(
        total=config.train_epochs,
        desc='Training',
        unit='epoch',
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {rate_fmt}{postfix} [{elapsed}<{remaining}]',
    )

    metrics = defaultdict(int)
    global_step = 0
    quantiles = torch.tensor(config.quantiles).to(device)

    for epoch in range(config.train_epochs):
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

            for i, q in enumerate(config.quantiles):
                metrics[f'train_loss_quantile_{q:.2f}'] += loss[i].item()

            metrics['train_loss_total'] += mean_loss.item()
            metrics['train_mae'] += torch.abs(y_batch - y_pred[..., 1:2]).mean().item()

            if global_step != 0 and global_step % config.log_freq == 0:
                wandb.log({k: v / config.log_freq for k, v in metrics.items()})

                metrics = defaultdict(int)

            overall_pbar.update(1 / steps_per_epoch)
            global_step += 1

        val_metrics = defaultdict(int)

        for X_val_batch, y_val_batch in tqdm(
            test_loader,
            desc=f'Test Epoch {epoch + 1}',
            leave=False,
            unit='step',
        ):
            y_val_pred = model(X_val_batch)

            val_loss = quantile_loss(y_val_batch, y_val_pred, quantiles)

            for i, q in enumerate(config.quantiles):
                val_metrics[f'val_loss_quantile_{q:.2f}'] += val_loss[i].item()
            val_metrics['val_total_loss'] += val_loss.mean().item()
            val_metrics['val_mae'] += (
                torch.abs(y_val_batch - y_val_pred[..., 1:2]).mean().item()
            )

        wandb.log({k: v / len(test_loader) for k, v in val_metrics.items()})

    overall_pbar.close()


if __name__ == '__main__':
    main()
