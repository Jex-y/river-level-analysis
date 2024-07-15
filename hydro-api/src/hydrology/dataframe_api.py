from hashlib import sha256
from pathlib import Path
from abc import ABC
from datetime import datetime, timedelta
import httpx
from typing import Callable
import polars as pl
from io import StringIO
import logging

__all__ = ['DataFrameApi']

log = logging.getLogger('hydrology')


class DataFrameApi(ABC):
    """Fetching DataFrames from an API."""

    client: httpx.Client
    cache_dir: Path
    cache_max_age: timedelta

    def __init__(
        self,
        api_base_url: httpx.URL,
        cache_dir: Path = Path('./.cache'),
        cache_max_age: timedelta = timedelta(minutes=1),
        http_client: httpx.Client = httpx.Client(),
    ):
        self.api_base_url = api_base_url
        self.cache_dir = cache_dir
        self.cache_max_age = cache_max_age
        self.http_client = http_client

    # def __del__(self):
    # self.http_client.close()

    @staticmethod
    def _cache_dataframe_load(
        load_func: Callable[..., pl.DataFrame],
    ) -> Callable[..., pl.LazyFrame]:
        def wrapper(
            self: 'DataFrameApi',
            *args,
            **kwargs,
        ) -> pl.LazyFrame:
            key = f'{type(self).__name__}__{load_func.__name__}__{args}__{kwargs}'
            key = sha256(key.encode()).hexdigest()
            filepath = self.cache_dir / f'{key}.parquet'

            if not self.cache_dir.exists():
                self.cache_dir.mkdir()

            if filepath.exists():
                last_modified = datetime.fromtimestamp(filepath.stat().st_mtime)
                if last_modified + self.cache_max_age > datetime.now():
                    log.info('Loading response from cache')
                    return pl.scan_parquet(filepath)
                else:
                    log.info('Response in cache is stale')
                    filepath.unlink()

            df = load_func(self, *args, **kwargs)

            assert (
                isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame)
            ), f'Expected DataFrame fuction {load_func.__name__} to return a pl.DataFrame or pl.LazyFrame, got {type(df)}'

            is_lazy = isinstance(df, pl.LazyFrame)

            log.info('Writing response to cache')
            (df.collect() if is_lazy else df).write_parquet(filepath)

            return df if is_lazy else df.lazy()

        return wrapper

    def _parse_response(self, response: httpx.Response) -> pl.DataFrame:
        response.raise_for_status()

        reponse_type = response.headers.get('content-type')

        match reponse_type.split(';')[0]:
            case 'text/csv':
                log.info('Reading CSV response')
                return pl.read_csv(StringIO(response.text))

            case 'application/json':
                log.warning(
                    f'A JSON response was received, but CSV is prefered. Request: {args}, {kwargs}'
                )
                return pl.DataFrame(response.json()['items'])

            case _:
                log.error(f'Unsupported response type: {reponse_type}')
                raise ValueError(f'Unsupported response type: {reponse_type}')

    @_cache_dataframe_load
    def _get(
        self,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        logging.info(f'get request: {args}, {kwargs}')

        return self._parse_response(self.http_client.get(*args, **kwargs))
