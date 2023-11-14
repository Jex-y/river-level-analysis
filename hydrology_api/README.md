# Hydrology API

**This package was build for internal use, please feel free to use it but it may have bugs**

This package is designed to read data from the UK government [Hydrology API](https://environment.data.gov.uk/hydrology/doc/reference).
The API provides:

-   River Level, Flow and Water Quality Data
-   Rainfall Data
-   Groundwater Level Data

This package currently only provides access to the River Level, Flow, and Rainfall.

## Example

```python
from hydrology_api import HydrologyAPI, Measure, process_hydrology_data

api = HydrologyApi(max_threads=5)

stations = api.get_stations_on_river('River Wear', Measure.LEVEL)
print(f"Loading data for {len(stations)} stations: {stations['label'].values}")
level_df = api.load(Measure.LEVEL, stations)
level_df = process_hydrology_data(level_df)
level_df.info()
>>> Loading data for 5 stations: ['Chester Le Street' 'Witton Park' 'Sunderland Bridge' 'Stanhope'
 'Durham New Elvet Bridge']
>>> Loading from cache: https://environment.data.gov.uk/hydrology/data/batch-readings/batch/?measure=e7d8bbb6-5bba-4057-9f49-a299482c3348-level-i-900-m-qualified&mineq-date=2007-01-01
>>> Loading from cache: https://environment.data.gov.uk/hydrology/data/batch-readings/batch/?measure=05784319-693a-4d75-b29e-32f01a99ee4f-level-i-900-m-qualified&mineq-date=2007-01-01
>>> Loading from cache: https://environment.data.gov.uk/hydrology/data/batch-readings/batch/?measure=ddedb4d9-b2be-47c1-998d-acbc0ffb124b-level-i-900-m-qualified&mineq-date=2007-01-01
>>> Loading from cache: https://environment.data.gov.uk/hydrology/data/batch-readings/batch/?measure=b29c481a-5012-40f5-bb0c-f9370be34975-level-i-900-m-qualified&mineq-date=2007-01-01
>>> Loading from cache: https://environment.data.gov.uk/hydrology/data/batch-readings/batch/?measure=ba3f8598-e654-430d-9bb8-e1652e6ff93d-level-i-900-m-qualified&mineq-date=2007-01-01
>>> <class 'pandas.core.frame.DataFrame'>
>>> DatetimeIndex: 590967 entries, 2007-01-01 00:00:00 to 2023-11-08 21:30:00
>>> Freq: 15T
>>> Data columns (total 5 columns):
>>>  #   Column                   Non-Null Count   Dtype
>>> ---  ------                   --------------   -----
>>>  0   Chester Le Street        590967 non-null  float16
>>>  1   Durham New Elvet Bridge  563826 non-null  float16
>>>  2   Stanhope                 501615 non-null  float16
>>>  3   Sunderland Bridge        589418 non-null  float16
>>>  4   Witton Park              585167 non-null  float16
>>> dtypes: float16(5)
>>> memory usage: 10.1 MB
```
