from datetime import datetime, timedelta, timezone

import httpx
import polars as pl
from hydrology.flooding import FloodingApi


async def test_get_last_24_hours(stations: pl.DataFrame):
    async with httpx.AsyncClient() as http_client:
        api = FloodingApi(http_client)
        df = await api.get_last_n_measures(
            stations,
            24,
        )
        assert len(df) == 24
        assert df['dateTime'].dtype == pl.Datetime
        assert df['dateTime'].min() >= datetime.now(tz=timezone.utc) - timedelta(days=1)
        assert df['dateTime'].max() <= datetime.now(tz=timezone.utc)

        expected_cols = set(
            stations.select(
                pl.format('{} - {}', pl.col('label'), pl.col('parameter'))
            ).to_series()
        ) | {'dateTime'}

        assert set(df.columns) == expected_cols
