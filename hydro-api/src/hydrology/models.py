from enum import Enum
from polars.datatypes import Enum as PolarsEnum


class Parameter(Enum):
    """Measurement type enum."""

    LEVEL = 'level'
    """Water level measurements."""

    RAINFALL = 'rainfall'
    """Rainfall measurements."""


ParameterEnumPolars = PolarsEnum([param.value for param in Parameter])
