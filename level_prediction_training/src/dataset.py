import math
from datetime import datetime, timedelta

import httpx
import polars as pl
import torch
from hydrology import HydrologyApi
from torch.utils.data import Dataset


def calculate_time_features(
    datetime_series: pl.Series,
):
    start_year = 2007
    return (
        datetime_series.to_frame()
        .with_columns(
            (datetime_series.dt.ordinal_day() / 365).alias('day_of_year'),
            (
                (
                    datetime_series.dt.epoch(time_unit='s')
                    - datetime(start_year, 1, 1).timestamp()
                )
                / (60 * 60 * 24 * 365.2524)
            ).alias(f'years_since_{start_year}'),
        )
        .select(
            (pl.col('day_of_year') * 2 * math.pi).cos().alias('cos_day_of_year'),
            (pl.col('day_of_year') * 2 * math.pi).sin().alias('sin_day_of_year'),
            pl.col(f'years_since_{start_year}'),
        )
    )


def load_training_data(
    stations: pl.DataFrame,
    train_split: float = 0.8,
):
    with httpx.Client() as http_client:
        api = HydrologyApi(http_client, cache_max_age=timedelta(weeks=1))

        df = api.get_measures(stations, start_date=datetime(2007, 1, 1))

    df = pl.concat(
        [df, calculate_time_features(df['dateTime'])], how='horizontal'
    ).drop('dateTime')

    train_records = int(len(df) * train_split)

    return df.slice(0, train_records), df.slice(train_records)


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
