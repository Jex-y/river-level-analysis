from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from io import StringIO
import urllib3
import ujson as json
import pandas as pd
import numpy as np
import os


class Measure(Enum):
    LEVEL = 'level'
    FLOW = 'flow'
    RAINFALL = 'rainfall'


class HydrologyApi:
    API_BASE_URL = "https://environment.data.gov.uk/hydrology/"
    DATA_DIR = "data"
    CACHE_DIR = "cache"

    units = {
        Measure.LEVEL: 'i-900-m-qualified',
        Measure.FLOW: 'i-900-m3s-qualified',
        Measure.RAINFALL: 't-900-mm-qualified',
    }

    observed_property_names = {
        Measure.LEVEL: 'waterLevel',
        Measure.FLOW: 'waterFlow',
        Measure.RAINFALL: 'rainfall',
    }

    def __init__(self, max_threads):
        self.http = urllib3.PoolManager(maxsize=max_threads)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)

    def get_stations_on_river(self, river: str, measure: Measure, limit=1000):
        api_url = self.API_BASE_URL + 'id/stations'
        result = self.http.request(
            'GET',
            api_url,
            fields={
                'observedProperty': HydrologyApi.observed_property_names[measure],
                'riverName': river,
                '_limit': limit,
                "status.label": "Active"
            }
        ).data.decode('utf-8')
        data = json.loads(result)
        return pd.DataFrame(data['items'])

    def get_stations_close_to_with_measure(self, lat, lon, radius, measure: Measure, limit=100):
        api_url = self.API_BASE_URL + 'id/stations'

        result = self.http.request(
            'GET',
            api_url,
            fields={
                'observedProperty': measure.value,
                'lat': lat,
                'long': lon,
                'dist': radius,
                'status.label': 'Active',
                '_limit': limit
            }
        ).data.decode('utf-8')
        data = json.loads(result)
        return pd.DataFrame(data['items'])

    def get_measure(self, measure: Measure, station_id: str, start=None):
        api_url = self.API_BASE_URL + \
            f"id/measures/{station_id}-{measure.value}-{HydrologyApi.units[measure]}/readings"
        # result = urlopen(api_url).read().decode('utf-8')
        result = self.http.request(
            'GET',
            api_url,
            fields={}
            | ({
                'mineq-date': start.strftime('%Y-%m-%d')
            } if start is not None else {}),
        ).data.decode('utf-8')
        data = json.loads(result)
        return pd.DataFrame(data['items'])

    def _cached_batch_request(self, api_url):
        cache_key = api_url.split('?')[1]
        cache_key = cache_key.replace('=', '_').replace('&', '_')
        filepath = os.path.join(self.CACHE_DIR, f"{cache_key}.feather")

        if os.path.exists(filepath):
            print(f"Loading from cache: {api_url}")

            with open(filepath, 'rb') as f:
                return pd.read_feather(f)

        data = self._batch_request(api_url)

        if data is not None:
            if not os.path.exists(self.CACHE_DIR):
                os.makedirs(self.CACHE_DIR)

            data.to_feather(filepath)

        return data

    def _batch_request(self, api_url):
        status = "Pending"

        while status in ("Pending", "InProgress"):
            print(f"Making request to: {api_url}")

            request = self.http.request(
                'GET',
                api_url,
                headers={
                    'Accept-Encoding': 'gzip'
                }
            )
            content_type = request.headers['Content-Type']

            if content_type == 'text/csv':
                if len(request.data) == 0:
                    print('Got empty CSV')
                    return None
                buffer = StringIO(request.data.decode('utf-8'))
                return pd.read_csv(buffer, low_memory=False)

            assert content_type in (
                'application/json',
                'application/json;charset=UTF-8'), f"Unexpected content type: {content_type}"

            data = json.loads(request.data.decode('utf-8'))
            status = data["status"]

            if status == "Pending":
                print(f"Query is pending")
                pos_in_queue = data["positionInQueue"]
                print(f"Position in queue: {pos_in_queue}")
                eta = data["eta"] / 1000
                print(f"Estimated completion: {eta}")
                sleep(eta * 1.1)

            elif status == "InProgress":
                print(f"Query in progress")
                eta = data["eta"] / 1000
                print(f"Estimated completion: {eta}")
                sleep(eta * 1.1)

            elif status in ("Complete", "Completed"):
                print(f"Query completed: {data}")
                csv_url = data["dataUrl"] if "dataUrl" in data else data["url"]
                return pd.read_csv(csv_url)

            elif status == "Failed":
                raise Exception(f"Query failed, response: {data}")

            else:
                raise Exception(f"Unknown status: {data['status']}")

    def batch_get_measure(self, measure: Measure, station_id):
        try:
            api_url = self.API_BASE_URL + \
                f"data/batch-readings/batch/?measure={station_id}-{measure.value}-{HydrologyApi.units[measure]}&mineq-date=2007-01-01"

            return self._cached_batch_request(api_url)
        except Exception as e:
            print(f"Failed to get data for station: {station_id}, {e}")
            return None

    def batch_get_measure_on_river(self, measure: Measure, river):
        stations = self.get_stations_on_river(river)
        return self.batch_get_measure_from_stations(measure, stations)

    def batch_get_measure_from_stations(self, measure: Measure, stations):
        data = pd.DataFrame()
        threads = [
            self.thread_pool.submit(
                self.batch_get_measure, measure, station_id)
            for station_id in stations['notation'].values
        ]

        for thread, station_name in zip(threads, stations['label'].values):
            new_data = thread.result()
            if new_data is None:
                print(f"No new data for station: {station_name}")
                continue
            new_data = new_data.drop(
                columns=['measure', 'date', 'qcode', 'completeness'])
            new_data['station'] = station_name
            new_data['station'] = new_data['station'].astype('category')
            new_data['dateTime'] = pd.to_datetime(new_data['dateTime'])
            new_data['value'] = new_data['value'].astype(float)
            new_data['quality'] = new_data['quality'].astype('category')
            data = pd.concat([data, new_data])
            data.drop_duplicates(subset=['dateTime', 'station'], inplace=True)
        return data

    def get_filename(self, measure: Measure, river):
        return f"{river.lower().replace(' ', '_')}_{measure.value}_raw.feather"

    def update_dataframe(self, df: pd.DataFrame, measure: Measure, stations: pd.DataFrame):
        for station_name, station_id in stations[['label', 'notation']].values:

            last = df[df['station'] == station_name]['dateTime'].max() if len(
                df) > 0 else None

            new_measurements = self.get_measure(measure, station_id, last)[
                ['dateTime', 'value', 'quality']]

            new_measurements['station'] = station_name
            new_measurements['station'] = new_measurements['station'].astype(
                'category')
            new_measurements['dateTime'] = pd.to_datetime(
                new_measurements['dateTime'])
            new_measurements['value'] = new_measurements['value'].astype(float)

            df = pd.concat([df, new_measurements])

        df.drop_duplicates(subset=['dateTime', 'station'], inplace=True)
        return df

    def load(self, measure: Measure, stations: pd.DataFrame):
        df = self.batch_get_measure_from_stations(measure, stations)
        df = self.update_dataframe(df, measure, stations)
        return df


def process_hydrology_data(df):
    return df[df['quality'].isin(['Good', 'Unchecked', 'Estimated'])] \
        .pivot(index='dateTime', columns='station', values='value') \
        .resample('15min').interpolate('time', limit_direction='both', limit=24*4, fill_value='extrapolate') \
        .astype(np.float16)
