import logging
from datetime import datetime, timedelta

import httpx
import polars as pl
import torch
from hydrology import HydrologyApi
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .config import Config


def load_training_data(
    stations: pl.DataFrame,
    train_split: float = 0.8,
):
    with httpx.Client() as http_client:
        api = HydrologyApi(http_client, cache_max_age=timedelta(weeks=1))

        df = api.get_measures(stations, start_date=datetime(2007, 1, 1)).rename(
            {"dateTime": "datetime"}
        )

    train_records = int(len(df) * train_split)

    return df.slice(0, train_records), df.slice(train_records)


class DataModule(LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        log = logging.getLogger("training")

        self.stations = pl.read_json(config.stations_filepath)

        log.info(f"Loaded {len(self.stations)} stations")

        train_df, test_df = load_training_data(
            self.stations, train_split=config.train_split
        )

        log.info(
            f"Loaded {len(train_df)} training samples and {len(test_df)} test samples"
        )

        # Sort columns alphabetically to ensure consistent order for inference
        self.x_column_names = sorted(list(train_df.drop("datetime").columns))
        train_df = train_df.select(["datetime", *self.x_column_names])
        test_df = test_df.select(["datetime", *self.x_column_names])
        self.stations = self.stations.sort("label")

        # This doesn't deal with the case where a station measures multiple parameters
        assert len(self.stations["label"].unique()) == len(self.stations), (
            "Each station should have a unique label. "
            "Stations measured multiple parameters are not currently supported."
        )

        # For datetime, calculate integer columns for day of year and year
        x_train_datetime = train_df.select(
            pl.col("datetime").dt.ordinal_day().alias("day_of_year").cast(pl.Int32),
            pl.col("datetime").dt.year().alias("year").cast(pl.Int32),
        ).to_torch()

        x_train = train_df.drop("datetime").to_torch().type(torch.float32)
        y_train = train_df.select(config.target_col).to_torch().type(torch.float32)

        x_test_datetime = test_df.select(
            (pl.col("datetime").dt.ordinal_day() - 1)
            .alias("day_of_year")
            .cast(pl.Int32),
            pl.col("datetime").dt.year().alias("year").cast(pl.Int32),
        ).to_torch()
        x_test = test_df.drop("datetime").to_torch().type(torch.float32)
        y_test = test_df.select(config.target_col).to_torch().type(torch.float32)

        x_len = max(*config.rolling_windows, config.context_length)

        self.train_dataset = TimeSeriesDataset(
            x_train_datetime,
            x_train,
            y_train,
            x_len,
            config.prediction_length,
        )
        self.test_dataset = TimeSeriesDataset(
            x_test_datetime,
            x_test,
            y_test,
            x_len,
            config.prediction_length,
        )

    @property
    def num_features(self):
        return self.train_dataset.x.shape[1]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        x_datetime: list[datetime],
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        x_len: int,
        y_len: int,
    ):
        self.x_datetime = x_datetime
        self.x = x
        self.y = y
        self.x_len = x_len
        self.y_len = y_len

    def __len__(self):
        return len(self.x) - self.x_len - self.y_len + 1

    def __getitem__(self, idx):
        # returned datetime should correspond to the last element in the context window
        return (
            self.x_datetime[idx + self.x_len - 1],
            self.x[idx : idx + self.x_len],
            self.y[idx + self.x_len : idx + self.x_len + self.y_len],
        )
