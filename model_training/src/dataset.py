import math
from datetime import datetime

import polars as pl
import numpy as np
from torch.utils.data import Dataset
from hydrology import HydrologyApi, Measure


def load_data(
    train_split: float = 0.8,
):
    api = HydrologyApi()

    level_stations = api.get_stations(
        Measure.MeasureType.LEVEL, river='River Wear'
    ).collect()

    rainfall_stations = (
        api.get_stations(
            Measure.MeasureType.RAINFALL, position=(54.774, -1.558), radius=15
        )
        .filter(
            ~pl.col('station_name').is_in(
                # Stations with lots of missing data
                [
                    'ESH Winning',
                    'Stanley Hustledown',
                    'Washington',
                ]
            )
        )
        .collect()
    )

    measures = [
        Measure(station_id, Measure.MeasureType.LEVEL)
        for station_id in level_stations['station_id']
    ] + [
        Measure(station_id, Measure.MeasureType.RAINFALL)
        for station_id in rainfall_stations['station_id']
    ]

    stations = pl.concat(
        [
            level_stations,
            rainfall_stations,
        ],
    ).unique()

    df = api.get_measures(measures, stations, start_date=datetime(2007, 1, 1))

    start_year = 2007

    time_features = (
        df.select(pl.col('timestamp'))
        .with_columns(
            (pl.col('timestamp').dt.ordinal_day() / 365).alias('day_of_year'),
            (
                (
                    pl.col('timestamp').dt.epoch(time_unit='s')
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

    df = pl.concat([df, time_features], how='horizontal')

    train_records = int(len(df) * train_split)

    return df.slice(0, train_records), df.slice(train_records)


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X: np.array,
        y: np.array,
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
