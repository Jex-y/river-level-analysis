from enum import Enum
from .dataframe_api import DataFrameApi
from datetime import datetime, timedelta
from io import StringIO
import httpx
import polars as pl
from time import sleep
from typing import Tuple
from .models import Parameter, ParameterEnumPolars
from logging import getLogger
from .utils import remove_none

logger = getLogger('hydrology')


class HydrologyApi(DataFrameApi):
    """The hydrology API provides river level, river flow, rainfall ect.

    API documentation: https://environment.data.gov.uk/hydrology/doc/reference
    """

    def __init__(self, cache_max_age: timedelta = timedelta(minutes=1)):
        super().__init__(
            api_base_url=httpx.URL('https://environment.data.gov.uk/hydrology/'),
            cache_max_age=cache_max_age,
        )

    @DataFrameApi._cache_dataframe_load
    def _get_batch(
        self,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        """Deal with batch requests from the API. These may be queued for some time before returning data.

        Returns:
            pd.DataFrame: The data returned by the API
        """
        logger.info(f'Batch request: {args}, {kwargs}')

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

        kwargs['headers'] = {**kwargs.get('headers', {}), **{'Accept-Encoding': 'gzip'}}

        while status in [BatchRequestStatus.PENDING, BatchRequestStatus.IN_PROGRESS]:
            response = self.http_client.get(*args, **kwargs, follow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', None)

            if content_type == 'text/csv':
                logger.debug('Received immediate response as CSV')
                return pl.read_csv(StringIO(response.text))

            assert (
                content_type is not None and 'application/json' in content_type
            ), f'Unexpected content type: {content_type}. Response: {response.text}'

            response_data: dict = response.json()

            assert 'status' in response_data, 'No status field in response'
            status = BatchRequestStatus.from_string(response_data['status'])

            match status:
                case BatchRequestStatus.PENDING | BatchRequestStatus.IN_PROGRESS:
                    eta = response_data.get('eta', 60 * 1000) / 1000
                    logger.info(f'Batch request status: {status}. ETA: {eta}')
                    logger.debug(f'Response data: {response_data}')
                    sleep(max(min(eta, 10), 1))

                case BatchRequestStatus.COMPLETE:
                    logger.info('Batch request complete')

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

                    logger.info(f'Fetching data from: {data_url}')
                    return pl.read_csv(data_url)

                case BatchRequestStatus.FAILED:
                    logger.error(f'Batch request failed: {response_data}')
                    raise Exception(f'Batch request failed: {response_data}')

                case _:
                    logger.error(f'Batch request unknown status: {status}')
                    raise Exception(f'Unknown status: {status}')

    @DataFrameApi._cache_dataframe_load
    def get_stations(
        self,
        measures: Parameter | list[Parameter] | None = None,
        river: str = None,
        position: Tuple[float, float] = None,
        radius: float = None,
        limit: int = None,
    ) -> pl.DataFrame:
        if isinstance(measures, Parameter):
            measures = [measures]

        lat, long = position if position else (None, None)

        parameter_observed_property = {
            Parameter.LEVEL: 'waterLevel',
            Parameter.RAINFALL: 'rainfall',
        }

        result = self.http_client.get(
            self.api_base_url.join('id/stations.csv'),
            params=remove_none(
                {
                    'observedProperty': [
                        parameter_observed_property[measure] for measure in measures
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

        result.raise_for_status()
        return pl.read_csv(
            StringIO(result.text),
            columns=['hydrology_api_notation', 'label', 'RLOIid', 'lat', 'long'],
            schema_overrides={
                'RLOIid': pl.Utf8,  # Sometimes gets interpreted as an int
            },
        )

    def _encode_measure(
        self,
        station_guid: str,
        parameter: Parameter,
    ):
        parameter = Parameter(parameter)

        units = {
            Parameter.LEVEL: 'level-i-900-m-qualified',
            Parameter.RAINFALL: 'rainfall-t-900-mm-qualified',
        }

        return f'{station_guid}-{units[parameter]}'

    def get_measures(
        self,
        stations: pl.DataFrame,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """Get measures for a list of stations between two dates.

        Args:
            stations (pl.DataFrame): A polars dataframe with columns 'hydrology_api_notation', 'label', and 'parameter'.
            start_date (datetime): The start date to get measures from.
            end_date (datetime | None, optional): The end date to get measures up to. Defaults to include up to the most recent data.

        Returns:
            pl.DataFrame: A polars dataframe with columns...
        """

        assert (
            'hydrology_api_notation' in stations.columns
        ), 'Stations dataframe must have a column "notation"'
        assert (
            'label' in stations.columns
        ), 'Stations dataframe must have a column "label"'
        assert (
            'parameter' in stations.columns
        ), 'Stations dataframe must have a column "parameter"'

        if isinstance(stations['parameter'], pl.List):
            stations = stations.explode('parameter')

        stations = stations.with_columns(
            pl.col('parameter').cast(ParameterEnumPolars),
        )

        # Estimate how many rows we are going to get back
        # Each measure is every 15 mins
        estimated_rows = (
            4 * 24 * (end_date or datetime.now() - start_date).days * len(stations)
        )

        logger.debug(f'Estimated rows to be fetched: {estimated_rows}')

        params = remove_none(
            {
                'measure': [
                    self._encode_measure(*measure)
                    for measure in stations[
                        ['hydrology_api_notation', 'parameter']
                    ].iter_rows()
                ],
                'mineq-date': start_date.strftime('%Y-%m-%d'),
                'maxeq-date': end_date.strftime('%Y-%m-%d') if end_date else None,
            }
        )

        if estimated_rows > 2_000_000:
            # We need to use the batch api
            logger.debug('Using batch API as estimated rows > 2,000,000')
            df = self._get_batch(
                self.api_base_url.join('data/batch-readings/batch'),
                params=params,
            )

        else:
            logger.debug('Using standard API as estimated rows < 2,000,000')
            df = self._get(
                self.api_base_url.join('data/readings.csv'),
                params=params,
            )

        with pl.StringCache():
            return (
                df.with_columns(
                    pl.col('measure').str.extract_groups(
                        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})-(\w+)-\w-\d+-(\w+)-qualified'
                    )
                )
                .select(
                    pl.col('value').cast(pl.Float32),
                    pl.col('quality').cast(pl.Categorical),
                    pl.col('dateTime').str.to_datetime(),
                    pl.col('measure').struct[0].alias('hydrology_api_notation'),
                    pl.col('measure')
                    .struct[1]
                    .alias('parameter')
                    .cast(ParameterEnumPolars),
                    pl.col('measure').struct[2].alias('units'),
                )
                .filter(pl.col('quality').is_in(['Good', 'Unchecked', 'Estimated']))
                .join(
                    stations if isinstance(stations, pl.LazyFrame) else stations.lazy(),
                    on=['hydrology_api_notation', 'parameter'],
                    how='inner',
                )
                .select(
                    pl.format(
                        '{} - {}',
                        pl.col('label'),
                        pl.col('parameter'),
                    ).alias('label'),
                    pl.col('dateTime'),
                    pl.col('value'),
                )
                .collect()  # Pivot can't be lazy
                .pivot(
                    'label',
                    index='dateTime',
                    values='value',
                )
                .sort('dateTime')
                .upsample(time_column='dateTime', every='15m')
                .interpolate()
                .fill_null(strategy='forward')
            )
