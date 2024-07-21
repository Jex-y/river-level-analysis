import asyncio
import logging
from datetime import UTC, datetime

import httpx
import polars as pl
from firebase_admin import storage

from .db import get_weather_forecasts_collection

log = logging.getLogger("weather_forecast")

http_client = httpx.AsyncClient()


def get_api_key():
    from .utils import get_dotenv

    return get_dotenv()["TOMORROW_API_KEY"]

    # from firebase_functions.params import StringParam

    # return StringParam('TOMORROW_API_KEY').value


async def get_weather_forecast(get_api_key):
    API_BASE_URL = httpx.URL("https://api.tomorrow.io/v4/weather/forecast")

    lat, long = 54.774, -1.558

    response = await http_client.get(
        API_BASE_URL,
        params={
            "location": f"{lat}, {long}",
            "timesteps": "1h",
            "units": "metric",
            "apikey": get_api_key(),
        },
        headers={"Accept": "application/json"},
    )

    response.raise_for_status()
    response_data = response.json()

    return (
        (
            pl.DataFrame(response_data["timelines"]["hourly"])
            .unnest("values")
            .select(
                pl.col("time").cast(pl.Datetime),
                pl.col("precipitationProbability"),
                pl.col(
                    r"^(rain|sleet|snow|ice|freezingRain)(Accumulation|AccumulationLwe|Intensity)$"
                ),
                pl.col("temperature"),
                pl.col("temperatureApparent"),
                pl.col("uvIndex"),
                pl.col("visibility"),
                pl.col("windGust"),
                pl.col("windSpeed"),
            )
        ),
        {
            "location": response_data["location"],
        },
    )


async def insert_weather_forecast(df: pl.DataFrame, metadata: dict):
    collection = get_weather_forecasts_collection()

    # record id is location + forecast_timestamp

    record = {
        "forecast_timestamp": datetime.now(UTC),
        "forecast_data": df.to_dict(as_series=False),
        "metadata": metadata,
    }

    log.info("Inserting weather forecast data into the database")

    await asyncio.to_thread(collection.insert_one, record)

    log.info("Database insert complete")


async def write_latest_forecast_to_bucket(df: pl.DataFrame):
    log.info("Uploading latest forecast to bucket")

    blob = storage.bucket().blob("weather/latest_forecast.json")

    await asyncio.to_thread(blob.upload_from_string, df.write_json())

    await asyncio.to_thread(
        blob.make_public,
    )

    public_url = blob.public_url

    log.info(f"Uploaded latest forecast to {public_url}")


async def collect_weather_forecast():
    df, metadata = await get_weather_forecast(get_api_key)
    log.info(
        f'Loaded {len(df)} weather forecast records between {df["time"].min()} and {df["time"].max()}'
    )
    await asyncio.gather(
        insert_weather_forecast(df, metadata),
        write_latest_forecast_to_bucket(df),
    )
