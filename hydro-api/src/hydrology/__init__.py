from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Callable, List, Tuple

import httpx
import polars as pl

__all__ = ['HydrologyApi', 'Measure']


def remove_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class Measure:
    STATION_ID_REGEX = (
        r'/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
    )
    MEASURE_TYPE_REGEX = (
        r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-(.*)-'
    )

    class MeasureType(Enum):
        LEVEL = 'level'
        FLOW = 'flow'
        RAINFALL = 'rainfall'

        @property
        def units(self):
            try:
                return {
                    self.LEVEL.value: 'i-900-m-qualified',
                    self.FLOW.value: 'i-900-m3s-qualified',
                    self.RAINFALL.value: 't-900-mm-qualified',
                }[self.value]
            except KeyError:
                raise NotImplementedError(f'Unknown measure type: {self.value}')

        @property
        def observed_property_name(self):
            try:
                return {
                    self.LEVEL.value: 'waterLevel',
                    self.FLOW.value: 'waterFlow',
                    self.RAINFALL.value: 'rainfall',
                }[self.value]
            except KeyError:
                raise NotImplementedError(f'Unknown measure type: {self.value}')

        @property
        def enum_dtype(self):
            return pl.Enum([value for value in self.__class__])

    def __init__(
        self,
        station_id: str,
        measure_type: MeasureType,
    ):
        self.station_id = station_id
        self.measure_type = measure_type

    def __str__(self):
        return f'{self.station_id}-{self.measure_type.value}-{self.measure_type.units}'

    def __repr__(self):
        return (
            f'Measure(station_id={self.station_id}, measure_type={self.measure_type})'
        )


class HydrologyApi:
    API_BASE_URL = httpx.URL('https://environment.data.gov.uk/hydrology/')
    CACHE_DIR = Path('./.cache')
    START_DATE = datetime(2008, 1, 1)
    CACHE_MAX_AGE = timedelta(weeks=1)

    def __init__(self):
        self.client = httpx.Client(follow_redirects=True)

    def __del__(self):
        self.client.close()

    @staticmethod
    def _cache_dataframe_load(
        load_func: Callable[..., pl.DataFrame],
    ) -> Callable[..., pl.LazyFrame]:
        def wrapper(
            self,
            *args,
            **kwargs,
        ) -> pl.LazyFrame:
            key = f'{load_func.__name__}__{args}__{kwargs}'
            key = sha256(key.encode()).hexdigest()

            filepath = HydrologyApi.CACHE_DIR / f'{key}.parquet'

            if not HydrologyApi.CACHE_DIR.exists():
                HydrologyApi.CACHE_DIR.mkdir()

            if filepath.exists():
                last_modified = datetime.fromtimestamp(filepath.stat().st_mtime)
                if last_modified + HydrologyApi.CACHE_MAX_AGE > datetime.now():
                    return pl.scan_parquet(filepath)
                else:
                    filepath.unlink()

            df = load_func(self, *args, **kwargs)
            df.write_parquet(filepath)
            return df.lazy()

        return wrapper

    @_cache_dataframe_load
    def _batch_request(
        self,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        """Deal with batch requests from the API. These may be queued for some time before returning data.

        Returns:
            pd.DataFrame: The data returned by the API
        """

        class BatchRequestStatus(Enum):
            PENDING = 'pending'
            IN_PROGRESS = 'inprogress'
            COMPLETE = 'complete'
            FAILED = 'failed'

            @staticmethod
            def from_string(s: str):
                s = s.lower()
                s = 'complete' if s == 'completed' else s

                assert s in [
                    e.value for e in BatchRequestStatus
                ], f'Unknown response status: {s}'
                return BatchRequestStatus(s)

        status = BatchRequestStatus.PENDING

        required_headers = {'Accept-Encoding': 'gzip'}
        kwargs['headers'] = {**kwargs.get('headers', {}), **required_headers}

        while status in [BatchRequestStatus.PENDING, BatchRequestStatus.IN_PROGRESS]:
            response = self.client.get(*args, **kwargs)

            content_type = response.headers.get('content-type', None)

            if content_type == 'text/csv':
                buffer = StringIO(response.text)
                return pl.read_csv(buffer, low_memory=False)

            assert (
                'application/json' in content_type
            ), f'Unexpected content type: {content_type}'

            response_data: dict = response.json()
            assert 'status' in response_data, 'No status field in response'
            status = BatchRequestStatus.from_string(response_data['status'])

            match status:
                case BatchRequestStatus.PENDING | BatchRequestStatus.IN_PROGRESS:
                    eta = response_data.get('eta', 60 * 1000) / 1000
                    sleep(max(eta * 0.1, 1))

                case BatchRequestStatus.COMPLETE:
                    keys = [
                        'dataUrl',
                        'url',
                    ]  # Some responses have dataUrl, some have url
                    data_url = next(
                        (response_data.get(k) for k in keys if k in response_data), None
                    )
                    assert (
                        data_url
                    ), f'Could not find data URL in response: {response_data}'
                    return pl.read_csv(data_url)

                case BatchRequestStatus.FAILED:
                    raise Exception(f'Batch request failed: {response_data}')

                case _:
                    raise Exception(f'Unknown status: {status}')

    @_cache_dataframe_load
    def _request(
        self,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        response = self.client.get(*args, **kwargs)
        response.raise_for_status()

        response_data = response.json()
        assert 'items' in response_data, 'No items field in response'
        return pl.DataFrame(response_data['items'])

    @_cache_dataframe_load
    def get_stations(
        self,
        measures: Measure.MeasureType | List[Measure.MeasureType] | None = None,
        river: str = None,
        position: Tuple[float, float] = None,
        radius: float = None,
        limit: int = None,
    ) -> pl.DataFrame:
        if isinstance(measures, Measure.MeasureType):
            measures = [measures]

        lat, long = position if position else (None, None)

        result = self.client.get(
            HydrologyApi.API_BASE_URL.join('id/stations'),
            params=remove_none(
                {
                    'observedProperty': [
                        measure.observed_property_name for measure in measures
                    ]
                    if measures
                    else None,
                    'riverName': river,
                    'lat': lat,
                    'long': long,
                    'dist': radius,
                    '_limit': limit,
                    'status.label': 'Active',
                }
            ),
        )
        result_json = result.json()
        assert 'items' in result_json, f'Unexpected response: {result_json}'
        return pl.from_dicts(
            result_json['items'],
            schema={
                'label': pl.String,
                'stationGuid': pl.String,
                # "lat": pl.Float64,  # Not used atm
                # "long": pl.Float64,
                # "riverName": pl.String, # Breaks Polars for some reason
            },
        ).select(
            pl.col('stationGuid').alias('station_id'),
            pl.col('label').alias('station_name'),
        )

    def get_measures(
        self,
        measures: List[Measure],
        stations: pl.DataFrame,
        start_date: datetime = START_DATE,
    ) -> pl.DataFrame:
        # Estimate how many rows we are going to get back
        # Each measure is every 15 mins
        estimated_rows = 4 * 24 * (datetime.now() - start_date).days * len(measures)

        params = {
            'measure': [str(m) for m in measures],
            'mineq-date': start_date.strftime('%Y-%m-%d'),
            '_limit': int(estimated_rows * 1.1),
        }

        if estimated_rows > 2_000_000:
            # We need to use the batch api
            df = self._batch_request(
                HydrologyApi.API_BASE_URL.join('data/batch-readings/batch'),
                params=params,
            )

        else:
            df = self._request(
                HydrologyApi.API_BASE_URL.join('data/readings.json'), params=params
            ).with_columns(
                pl.col('measure').struct.field('@id').alias('measure'),
            )

        with pl.StringCache():
            return (
                df.select(
                    pl.col('value').cast(pl.Float32),
                    pl.col('quality').cast(pl.Categorical),
                    pl.col('dateTime').str.to_datetime().alias('timestamp'),
                    pl.col('measure'),  # .cast(pl.Categorical),
                )
                .filter(pl.col('quality').is_in(['Good', 'Unchecked', 'Estimated']))
                .with_columns(
                    pl.col('measure')
                    .str.extract(Measure.STATION_ID_REGEX)
                    .cast(pl.Categorical)
                    .alias('station_id'),
                    pl.col('measure')
                    .str.extract(Measure.MEASURE_TYPE_REGEX)
                    .cast(pl.Categorical)
                    .alias('measure_type'),
                )
                .join(
                    stations.select(
                        pl.col('station_id').cast(pl.Categorical),
                        pl.col('station_name').cast(pl.Categorical),
                    ).lazy(),
                    on='station_id',
                    how='inner',
                )
                .with_columns(
                    pl.format(
                        '{} {}', pl.col('station_name'), pl.col('measure_type')
                    ).alias('series_name'),
                )
                .collect()  # Pivot can't be lazy
                .pivot(
                    'series_name',
                    index='timestamp',
                    values='value',
                )
                .sort('timestamp')
                .upsample(time_column='timestamp', every='15m')
                .interpolate()
                .fill_null(strategy='forward')
            )
