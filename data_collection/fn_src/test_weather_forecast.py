from .weather_forecast import get_weather_forecast
import polars as pl
from datetime import timedelta
from .utils import get_dotenv


def get_api_key():
    return get_dotenv()["TOMORROW_API_KEY"]


async def test_get_weather_forecast():
    forecast, metadata = await get_weather_forecast(get_api_key)

    assert isinstance(forecast, pl.DataFrame)

    assert forecast["time"].dtype == pl.Datetime
    # Check that it is hourly
    assert (
        forecast["time"] - forecast["time"].shift(1)
    ).drop_nulls().unique().item() == timedelta(hours=1)

    # == timedelta(hours=1)

    # Check that is has at least these columns
    expected_columns = [
        "time",
        "precipitationProbability",
        "rainAccumulation",
        "rainIntensity",
        "temperature",
        "temperatureApparent",
        "uvIndex",
        "visibility",
        "windGust",
        "windSpeed",
    ]

    assert set(forecast.columns) >= set(expected_columns)
