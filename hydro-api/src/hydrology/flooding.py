from .dataframe_api import DataFrameApi
from .models import Parameter

from datetime import datetime
import polars as pl
import httpx
from .utils import remove_none
from io import StringIO
import asyncio


class FloodingApi(DataFrameApi):
    """Provides similar data to the HydrologyApi, but with a lower latency.
    This is useful for fetching the latest data, the HydrologyApi should be used for fetching large volumes of data.

    Methods are async. Responses are not cached to ensure the latest data is always fetched.
    """

    http_client: httpx.AsyncClient

    def __init__(self):
        super().__init__(
            api_base_url=httpx.URL('https://environment.data.gov.uk/flood-monitoring/'),
            http_client=httpx.AsyncClient(),
        )

    async def get_stations(
        self,
        measures: Parameter | list[Parameter] | None = None,
        river: str | None = None,
        position: tuple[float, float] | None = None,
        radius: float | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        if isinstance(measures, Parameter):
            measures = [measures]

        lat, long = position if position else (None, None)

        result = await self.http_client.get(
            self.api_base_url.join('id/stations.csv'),
            params=remove_none(
                {
                    'parameter': [measure.value for measure in measures]
                    if measures
                    else None,
                    'riverName': river,
                    'lat': lat,
                    'long': long,
                    'dist': radius,
                    '_limit': limit,
                    # 'status': 'Active', # Broken for rainfall stations :(
                }
            ),
        )

        result.raise_for_status()
        return pl.read_csv(
            StringIO(result.text),
            columns=['notation', 'label', 'RLOIid', 'lat', 'long'],
            schema_overrides={
                'RLOIid': pl.Utf8,
                'notation': pl.Utf8,
            },
        )

    def _encode_measure(
        self,
        station_notation: str,
        parameter: Parameter,
    ) -> str:
        parameter = Parameter(parameter)

        units = {
            Parameter.LEVEL: 'level-stage-i-15_min-m',
            Parameter.RAINFALL: 'rainfall-tipping_bucket_raingauge-t-15_min-mm',
        }

        return f'{station_notation}-{units[parameter]}'

    async def get_single_measure_last_n(
        self,
        station_notation: str,
        parameter: Parameter,
        n: int,
    ) -> pl.DataFrame:
        response = await self.http_client.get(
            self.api_base_url.join(
                f'id/measures/{self._encode_measure(station_notation, parameter)}/readings'
            ),
            headers={'Accept': 'text/csv'},
            params={
                '_sorted': None,
                '_limit': n,
            },
        )

        df = self._parse_response(response).select(
            pl.col('dateTime').cast(pl.Datetime), pl.col('value').cast(pl.Float32)
        )

        assert len(df) == n, f'Expected {n} rows, got {len(df)}'
        return df

    async def get_latest_observation_timestamp(
        self,
        station_notation: str,
        parameter: Parameter,
    ) -> datetime:
        response = await self.http_client.get(
            self.api_base_url.join(
                f'id/measures/{self._encode_measure(station_notation, parameter)}/readings'
            ),
            params={
                'latest': None,
            },
            headers={'Accept': 'application/json'},
        )

        response.raise_for_status()
        datetime_string = response.json()['items'][0][
            'dateTime'
        ]  # Like 2024-07-15T13:30:00Z

        # If we are running in python 3.11, fromisoformat should handle the Z
        return datetime.fromisoformat(datetime_string)

    async def get_last_n_measures(
        self,
        stations: pl.DataFrame,
        n: int,
    ) -> pl.DataFrame:
        results = await asyncio.gather(
            *[
                self.get_single_measure_last_n(notation, parameter, n)
                for notation, parameter in stations[
                    ['flooding_api_notation', 'parameter']
                ].iter_rows()
            ]
        )

        return (
            pl.concat(
                [
                    df.rename({'value': f'{label} - {parameter}'})
                    for df, (label, parameter) in zip(
                        results,
                        stations[['label', 'parameter']].iter_rows(),
                    )
                ],
                how='align',
            )
            .sort('dateTime')
            .upsample(time_column='dateTime', every='15m')
            .interpolate()
            .fill_null(strategy='forward')
        )
