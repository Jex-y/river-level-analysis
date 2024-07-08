# Hydrology API

**This package was build for my own use, please feel free to use it but it may have bugs**

This package loads data from the UK [Hydrology API](https://environment.data.gov.uk/hydrology/doc/reference).
The API provides:

- River Level, Flow and Water Quality Data
- Rainfall Data
- Groundwater Level Data

This package currently only provides access to the River Level, Flow, and Rainfall. Data is returned as a polars DataFrame.
After the first request of a given dataset, the data is cached for up to a week.

## Example

```python
>>> from hydrology import HydrologyApi, Measure
>>> from datetime import datetime
>>> api = HydrologyApi()
>>> 
>>> # Get all stations that record the water level along the River Wear in Durham
>>> stations = api.get_stations(Measure.MeasureType.LEVEL, river="River Wear")
>>> 
>>> df = api.get_measures(
>>>     [
>>>       Measure(station_id, Measure.MeasureType.LEVEL) for station_id in stations["station_id"]
>>>     ],
>>>     stations,
>>>     start_date=datetime(2020, 1, 1),
>>> )
>>> 
>>> df.head()
shape: (5, 6)
┌────────────────┬────────────────┬────────────────┬───────────────┬───────────────┬───────────────┐
│ timestamp      ┆ Durham New     ┆ Sunderland     ┆ Chester Le    ┆ Witton Park   ┆ Stanhope      │
│ ---            ┆ Elvet Bridge   ┆ Bridge         ┆ Street        ┆ level-i-900-m ┆ level-i-900-m │
│ datetime[μs]   ┆ level-…        ┆ level-i-900-…  ┆ level-i-900-… ┆ ---           ┆ ---           │
│                ┆ ---            ┆ ---            ┆ ---           ┆ f32           ┆ f32           │
│                ┆ f32            ┆ f32            ┆ f32           ┆               ┆               │
╞════════════════╪════════════════╪════════════════╪═══════════════╪═══════════════╪═══════════════╡
│ 2020-01-01     ┆ 0.372          ┆ 0.417          ┆ 0.465         ┆ 0.465         ┆ 0.344         │
│ 00:00:00       ┆                ┆                ┆               ┆               ┆               │
│ 2020-01-01     ┆ 0.379          ┆ 0.418          ┆ 0.465         ┆ 0.465         ┆ 0.344         │
│ 00:15:00       ┆                ┆                ┆               ┆               ┆               │
│ 2020-01-01     ┆ 0.378          ┆ 0.417          ┆ 0.464         ┆ 0.465         ┆ 0.344         │
│ 00:30:00       ┆                ┆                ┆               ┆               ┆               │
│ 2020-01-01     ┆ 0.371          ┆ 0.417          ┆ 0.464         ┆ 0.465         ┆ 0.344         │
│ 00:45:00       ┆                ┆                ┆               ┆               ┆               │
│ 2020-01-01     ┆ 0.367          ┆ 0.417          ┆ 0.467         ┆ 0.465         ┆ 0.344         │
│ 01:00:00       ┆                ┆                ┆               ┆               ┆               │
└────────────────┴────────────────┴────────────────┴───────────────┴───────────────┴───────────────┘
```
