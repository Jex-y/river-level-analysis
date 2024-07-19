import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
from firebase_admin import initialize_app, storage
from firebase_functions import options, scheduler_fn

initialize_app(options={"storageBucket": "durham-river-level.appspot.com"})
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s"
)


@dataclass
class Config:
    context_length: int
    prediction_length: int
    target_col: str
    quantiles: list
    predict_difference: bool


@dataclass
class Metadata:
    last_observation: datetime
    last_prediction: datetime

    @staticmethod
    def from_json(json_str):
        metadata = json.loads(json_str)
        return Metadata(
            datetime.fromisoformat(metadata["last_observation"]),
            datetime.fromisoformat(metadata["last_prediction"]),
        )

    def to_json(self):
        return json.dumps(
            {
                "last_observation": self.last_observation.isoformat(),
                "last_prediction": self.last_prediction.isoformat(),
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


async def load_model() -> tuple:
    logging.info("Loading dependencies for model loading")

    from io import BytesIO
    from pickle import load as load_pickle

    from torch.jit import load as load_torchscript

    logging.info("Loading model from bucket")

    # Model is stored in the default bucket, in the model directory
    model_file_name = "model_torchscript.pt"
    preprocessing_file_name = "preprocessing.pickle"

    bucket = storage.bucket()

    model_buffer = bucket.blob(f"model/{model_file_name}")
    preprocessing_buffer = bucket.blob(f"model/{preprocessing_file_name}")

    model_buffer = download_blob_bytes(model_buffer)
    preprocessing_buffer = download_blob_bytes(preprocessing_buffer)

    model = load_torchscript(BytesIO(await model_buffer), map_location="cpu")
    preprocessing = load_pickle(BytesIO(await preprocessing_buffer))

    return model, preprocessing["X"], preprocessing["y"]


async def load_config() -> tuple[Config, pl.DataFrame]:
    logging.info("Loading config from bucket")
    from polars import DataFrame

    config_file_name = "inference_config.json"
    bucket = storage.bucket()
    config: dict = json.loads(
        await download_blob_bytes(bucket.blob(f"model/{config_file_name}"))
    )
    stations = DataFrame(config.pop("stations"))

    config = Config(**config)
    logging.debug(f"Loaded config: {config}")

    return config, stations


async def load_prev_metadata() -> Metadata:
    logging.info("Loading previous metadata from bucket")

    bucket = storage.bucket()
    blob = bucket.blob("prediction/medatadata.json")
    if not await asyncio.get_running_loop().run_in_executor(None, blob.exists):
        return None

    metadata = Metadata.from_json(await download_blob_bytes(blob))
    logging.debug(f"Loaded metadata: {metadata}")
    return metadata


async def save_metadata(metadata: Metadata):
    logging.info("Saving metadata to bucket")

    bucket = storage.bucket()
    blob = bucket.blob("prediction/medatadata.json")
    await upload_blob_string(blob, metadata.to_json(), content_type="application/json")


async def save_prediction(prediction: pl.DataFrame):
    logging.info("Saving prediction to bucket")

    bucket = storage.bucket()
    blob = bucket.blob("prediction/prediction.json")
    await upload_blob_string(
        blob, prediction.write_json(), content_type="application/json"
    )


async def save_observation(observation: pl.DataFrame):
    logging.info("Saving observation to bucket")

    bucket = storage.bucket()
    blob = bucket.blob("prediction/observation.json")
    await upload_blob_string(
        blob, observation.write_json(), content_type="application/json"
    )


def calculate_time_features(
    datetime_series,
):
    from polars import col

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
            (col("day_of_year") * 2 * math.pi).cos().alias("cos_day_of_year"),
            (col("day_of_year") * 2 * math.pi).sin().alias("sin_day_of_year"),
            col(f"years_since_{start_year}"),
        )
    )


async def predict(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    logging.info("Loading dependencies for prediction")
    from torch import tensor

    df = df.tail(config.context_length)
    df = pl.concat([df, calculate_time_features(df["dateTime"])], how="horizontal")

    model, X_preprocessing, y_preprocessing = await load_model()

    X = df.tail(config.context_length).drop("dateTime").to_pandas().astype("float32")
    X = tensor(X_preprocessing.transform(X)).unsqueeze(0)

    assert (
        X.shape[1] == config.context_length
    ), f"Expected at least {config.context_length} samples, got {X.shape[1]}"

    logging.info("Making prediction")

    y_pred = model(X).detach().numpy().reshape(-1, 1)
    y_pred = y_preprocessing.inverse_transform(y_pred).reshape(
        config.prediction_length, -1
    )

    if config.predict_difference:
        raise NotImplementedError("Predicting differences not implemented")
        # last_value = df[config.target_col][-1]
        # y_pred[:, 0] = y_pred[:, 0].cumsum(axis=0) + last_value

    last_observed = df["dateTime"].max()

    return pl.DataFrame(
        [
            pl.Series("predicted", y_pred[:, 0]),
            pl.Series("ci", y_pred[:, 1:]),
        ]
    ).with_columns(
        pl.datetime_range(
            last_observed,
            last_observed + timedelta(minutes=15 * config.prediction_length),
            interval="15m",
            closed="right",
        )
        .dt.timestamp()
        .alias("timestamp"),
    )


async def load_dataframe(config: Config, stations: pl.DataFrame):
    import httpx
    from hydrology import FloodingApi

    async with httpx.AsyncClient(timeout=httpx.Timeout(5)) as http_client:
        api = FloodingApi(http_client)

        # Might want to change this to return more data if the context length is short

        df = (
            (
                await api.get_last_n_measures(
                    stations,
                    max(config.context_length + 4, 24 * 4),
                )
            )
            .drop_nulls(subset=config.target_col)
            .with_columns(pl.exclude(config.target_col).forward_fill())
        )

    # Assert there are no nulls in the target column
    assert df[config.target_col].is_not_null().all(), "Null values in target colum"

    return df


async def run_inference(check_for_new_data: bool):
    logging.info("Loading config")

    config, stations = await load_config()
    prev_metadata = await load_prev_metadata()

    logging.info("Loading data")
    df = await load_dataframe(config, stations)

    # If the current last observation is within some threshold of the previous last observation, return

    latest_observation = df["dateTime"].max()
    if (
        check_for_new_data
        and prev_metadata
        and latest_observation == prev_metadata.last_observation
    ):
        logging.info("No new data available, last observation was over 15 minutes ago")
        return

    # Otherwise, make a prediction
    logging.info("Running model forward pass")

    # This blocks the main thread, but we should only be running the function once at a time so not an issue.

    prediction = await predict(df, config)

    await asyncio.gather(
        save_prediction(prediction),
        save_observation(
            df.tail(24 * 4).select(
                pl.col("dateTime").dt.timestamp().alias("timestamp"),
                pl.col(config.target_col).alias("observed"),
            )
        ),
        save_metadata(
            Metadata(
                latest_observation,
                latest_observation + timedelta(minutes=15 * config.prediction_length),
            )
        ),
    )


@scheduler_fn.on_schedule(
    schedule="*/5 * * * *", region="europe-west2", memory=options.MemoryOption.GB_1
)
def level_prediction_inference(_event):
    asyncio.run(run_inference(check_for_new_data=True))


if __name__ == "__main__":
    asyncio.run(run_inference(check_for_new_data=False))
