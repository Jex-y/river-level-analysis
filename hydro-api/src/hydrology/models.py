from dataclasses import dataclass
from enum import Enum
from polars.datatypes import Enum as PolarsEnum


@dataclass
class Station:
    """Measurement station dataclass.
    Used for both the historical and real time API.
    """

    station_name: str
    """Human readable name of the station."""

    rloi_id: int
    """River levels on the internet id. This uniquely identifies the station across both APIs.
    """


class Parameter(Enum):
    """Measurement type enum."""

    LEVEL = 'level'
    """Water level measurements."""

    RAINFALL = 'rainfall'
    """Rainfall measurements."""


ParameterEnumPolars = PolarsEnum([param.value for param in Parameter])
