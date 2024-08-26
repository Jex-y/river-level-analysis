import math
from datetime import datetime, timedelta

import httpx
import polars as pl
import torch
from hydrology import HydrologyApi
from torch.utils.data import Dataset
import logging
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import wandb


def calculate_time_features(
    datetime_series: pl.Series,
):
    start_year = 2007
    return (
        datetime_series.to_frame()
        .with_columns(
            (datetime_series.dt.ordinal_day() / 365).alias("day_of_year"),
            (
                (
                    datetime_series.dt.epoch(time_unit="s")
                    - datetime(start_year, 1, 1).timestamp()
                )
                / (60 * 60 * 24 * 365.2524)
            ).alias(f"years_since_{start_year}"),
        )
        .select(
            (pl.col("day_of_year") * 2 * math.pi).cos().alias("cos_day_of_year"),
            (pl.col("day_of_year") * 2 * math.pi).sin().alias("sin_day_of_year"),
            pl.col(f"years_since_{start_year}"),
        )
    )


def calculate_rolling_features(
    df,
    selector,
    windows=[7 * 24 * 4, 30 * 24 * 4],
):
    # Doesn't matter whether mean or sum is used as data is normalized later.
    return pl.concat(
        [
            df.select(selector.rolling_mean(window).name.suffix(f"_mean_{window}"))
            for window in windows
        ],
        how="horizontal",
    )


def load_training_data(
    stations: pl.DataFrame,
    train_split: float = 0.8,
):
    with httpx.Client() as http_client:
        api = HydrologyApi(http_client, cache_max_age=timedelta(weeks=1))

        df = api.get_measures(stations, start_date=datetime(2007, 1, 1))

    df = (
        pl.concat(
            [
                df,
                calculate_time_features(df["dateTime"]),
                calculate_rolling_features(
                    df,
                    (
                        pl.selectors.contains("rainfall (mm)")
                        | pl.selectors.contains("level (m)")
                    ),
                ),
            ],
            how="horizontal",
        )
        .drop("dateTime")
        .drop_nulls()
    )

    train_records = int(len(df) * train_split)

    return df.slice(0, train_records), df.slice(train_records)


class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        log = logging.getLogger("training")

        stations = pl.read_json(config.stations_filepath)

        log.info(f"Loaded {len(stations)} stations")
        wandb.log({"stations": stations["label"].to_list()})

        log.info("Loading training data")

        train_df, test_df = load_training_data(stations, train_split=config.train_split)

        log.info(
            f"Loaded {len(train_df)} training samples and {len(test_df)} test samples"
        )

        log.info("Preprocessing data")

        X_preprocessing = make_column_transformer(
            (
                QuantileTransformer(output_distribution="normal"),
                make_column_selector(pattern=r"- level \(m\)"),
            ),
            (
                QuantileTransformer(output_distribution="normal"),
                make_column_selector(pattern=r"- rainfall \(mm\)"),
            ),
            remainder="passthrough",
        )

        y_preprocessing = QuantileTransformer(output_distribution="normal")

        X_train = train_df.to_pandas().astype("float32")
        y_train = train_df.select(config.target_col).to_numpy().astype("float32")

        X_test = test_df.to_pandas().astype("float32")
        y_test = test_df.select(config.target_col).to_numpy().astype("float32")

        X_train = torch.tensor(X_preprocessing.fit_transform(X_train))
        y_train = torch.tensor(y_preprocessing.fit_transform(y_train))

        X_test = torch.tensor(X_preprocessing.transform(X_test))
        y_test = torch.tensor(y_preprocessing.transform(y_test))

        self.train_dataset = TimeSeriesDataset(
            X_train, y_train, config.sequence_length, config.prediction_length
        )
        self.test_dataset = TimeSeriesDataset(
            X_test, y_test, config.sequence_length, config.prediction_length
        )

    @property
    def num_features(self):
        return self.train_dataset.X.shape[1]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        seq_length: int,
        pred_length: int,
    ):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.X) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_length],
            self.y[idx + self.seq_length : idx + self.seq_length + self.pred_length],
        )
