import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from firebase_admin import initialize_app, storage
from firebase_functions import scheduler_fn
import logging
import asyncio


# Some imports are conditional to improve compute time efficiency when prediction is not required.

try:
    from google.auth import default as default_auth

    creds, _ = default_auth()
except:
    from firebase_admin import credentials
    from pathlib import Path

    creds = credentials.Certificate(
        Path(__file__).parent / '../firebase-service-account-key.json'
    )

initialize_app(
    credential=creds, options={'storageBucket': 'durham-river-level.appspot.com'}
)
logging.basicConfig(
    level=logging.INFO, format='[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
)


@dataclass
class Config:
    context_length: int
    prediction_length: int
    target_col: str
    quantiles: list


@dataclass
class Metadata:
    last_observation: datetime
    last_prediction: datetime

    @staticmethod
    def from_json(json_str):
        metadata = json.loads(json_str)
        return Metadata(
            datetime.fromisoformat(metadata['last_observation']),
            datetime.fromisoformat(metadata['last_prediction']),
        )

    def to_json(self):
        return json.dumps(
            {
                'last_observation': self.last_observation.isoformat(),
                'last_prediction': self.last_prediction.isoformat(),
            }
        )


async def download_blob_bytes(blob) -> bytes:
    return await asyncio.get_running_loop().run_in_executor(
        None, blob.download_as_bytes
    )


async def upload_blob_string(blob, data: bytes, content_type=None):
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: blob.upload_from_string(data, content_type=content_type)
    )


async def load_model():
    logging.info('Loading dependencies for model loading')

    from io import BytesIO
    from pickle import load as load_pickle

    from torch.jit import load as load_torchscript

    logging.info('Loading model from bucket')

    # Model is stored in the default bucket, in the model directory
    model_file_name = 'model_torchscript.pt'
    preprocessing_file_name = 'preprocessing.pickle'

    bucket = storage.bucket()

    model_buffer = bucket.blob(f'model/{model_file_name}')
    preprocessing_buffer = bucket.blob(f'model/{preprocessing_file_name}')

    model_buffer = download_blob_bytes(model_buffer)
    preprocessing_buffer = download_blob_bytes(preprocessing_buffer)

    model = load_torchscript(BytesIO(await model_buffer), map_location='cpu')
    preprocessing = load_pickle(BytesIO(await preprocessing_buffer))

    return model, preprocessing['X'], preprocessing['y']


async def load_config():
    logging.info('Loading config from bucket')
    from polars import DataFrame

    config_file_name = 'inference_config.json'
    bucket = storage.bucket()
    config: dict = json.loads(
        await download_blob_bytes(bucket.blob(f'model/{config_file_name}'))
    )
    stations = DataFrame(config.pop('stations'))

    config = Config(**config)
    logging.debug(f'Loaded config: {config}')

    return config, stations


async def load_prev_metadata():
    logging.info('Loading previous metadata from bucket')

    bucket = storage.bucket()
    blob = bucket.blob('prediction/medatadata.json')
    if not await asyncio.get_running_loop().run_in_executor(None, blob.exists):
        return None

    metadata = Metadata.from_json(await download_blob_bytes(blob))
    logging.debug(f'Loaded metadata: {metadata}')
    return metadata


async def save_metadata(metadata: Metadata):
    logging.info('Saving metadata to bucket')

    bucket = storage.bucket()
    blob = bucket.blob('prediction/medatadata.json')
    await upload_blob_string(blob, metadata.to_json(), content_type='application/json')


async def save_prediction(prediction):
    logging.info('Saving prediction to bucket')

    bucket = storage.bucket()
    blob = bucket.blob('prediction/prediction.json')
    await upload_blob_string(
        blob, json.dumps(prediction), content_type='application/json'
    )


def calculate_time_features(
    datetime_series,
):
    from polars import col

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
            (col('day_of_year') * 2 * math.pi).cos().alias('cos_day_of_year'),
            (col('day_of_year') * 2 * math.pi).sin().alias('sin_day_of_year'),
            col(f'years_since_{start_year}'),
        )
    )


async def predict(df, config: Config):
    logging.info('Loading dependencies for prediction')
    from torch import tensor

    model, X_preprocessing, y_preprocessing = await load_model()

    X = df.tail(config.context_length).drop('dateTime').to_pandas().astype('float32')
    X = tensor(X_preprocessing.transform(X)).unsqueeze(0)

    logging.info('Making prediction')

    assert X.shape[1] == config.context_length

    y_pred = model(X).detach().numpy().reshape(-1, 1)
    y_pred = y_preprocessing.inverse_transform(y_pred).reshape(
        config.prediction_length, len(config.quantiles)
    )

    assert config.quantiles[1] == 0.5

    # Predictions are at 15 minute intervals, starting from the last time in the input data
    y_pred_datetime = [
        df['dateTime'].max() + timedelta(minutes=15 * (i + 1))
        for i in range(config.prediction_length)
    ]

    return [
        {'timestamp': dateTime.isoformat(), 'value': float(value), 'type': 'observed'}
        for dateTime, value in df[['dateTime', config.target_col]].iter_rows()
    ] + [
        {
            'timestamp': dateTime.isoformat(),
            'value': float(values[1]),
            'type': 'predicted',
            'ci': [float(values[0]), float(values[2])],
        }
        for dateTime, values in zip(y_pred_datetime, y_pred)
    ]


# @scheduler_fn.on_schedule(schedule='*/5 * * * *', region='europe-west2')
async def predict_if_new_data():
    logging.info('Checking if new data is available')
    prev_metadata = await load_prev_metadata()

    # If the previous last observation was less than 15 minutes ago, return
    if prev_metadata and prev_metadata.last_observation > datetime.now() - timedelta(
        minutes=15
    ):
        logging.info(
            'No new data available, last observation was less than 15 minutes ago'
        )
        return

    from hydrology import FloodingApi

    api = FloodingApi()
    config, stations = await load_config()

    # Might want to change this to return more data if the context length is short

    import polars as pl

    df = (
        (
            await api.get_last_n_measures(
                stations,
                config.context_length + 4,  # Allow for up to an hours delay in data
            )
        )
        .drop_nulls(subset=config.target_col)
        .with_columns(pl.exclude(config.target_col).forward_fill())
    )

    # Assert there are no nulls in the target column
    assert df[config.target_col].is_not_null().all(), 'Null values in target colum'

    latest_observation = df['dateTime'].max()

    # If the current last observation is within some threshold of the previous last observation, return

    if prev_metadata and latest_observation == prev_metadata.last_observation:
        logging.info('No new data available, last observation was over 15 minutes ago')
        return

    # Otherwise, make a prediction
    logging.info('New data available, making prediction')

    # This blocks the main thread, but we should only be running the function once at a time so not an issue.

    df = pl.concat([df, calculate_time_features(df['dateTime'])], how='horizontal')

    prediction = await predict(df.tail(config.context_length), config)

    await asyncio.gather(
        save_prediction(prediction),
        save_metadata(
            Metadata(
                latest_observation,
                latest_observation + timedelta(minutes=15 * config.prediction_length),
            )
        ),
    )


if __name__ == '__main__':
    asyncio.run(predict_if_new_data())
