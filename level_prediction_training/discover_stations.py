from hydrology.hydrology import HydrologyApi
from hydrology.models import Parameter
import polars as pl
import pprint

api = HydrologyApi()

level_stations = (
    api.get_stations(Parameter.LEVEL, river='River Wear')
    .collect()
    .with_columns(pl.lit(Parameter.LEVEL).alias('parameter'))
)

rainfall_stations = (
    api.get_stations(Parameter.RAINFALL, position=(54.774, -1.558), radius=15)
    .filter(
        ~pl.col('label').is_in(
            # Stations with lots of missing data
            [
                'ESH Winning',
                'Stanley Hustledown',
                'Washington',
            ]
        )
    )
    .collect()
    .with_columns(pl.lit(Parameter.RAINFALL).alias('parameter'))
)

stations = pl.concat([level_stations, rainfall_stations])

print('Writing stations to stations.csv')
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
pprint.pprint(stations)

stations.write_csv('stations.csv')
