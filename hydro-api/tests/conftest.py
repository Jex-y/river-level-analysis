import polars as pl
import pytest

__all__ = [
    'stations',
]


@pytest.fixture()
def stations():
    return pl.DataFrame(
        [
            {
                'label': 'North Dalton',
                'hydrology_api_notation': '9fcbf8c6-b643-4f58-a2be-8beff3eda295',
                'parameter': 'rainfall',
                'flooding_api_notation': '025878',
            },
            {
                'label': 'Peterlee',
                'hydrology_api_notation': '935b389b-7ab8-46e6-9758-f8eb3861fbba',
                'parameter': 'rainfall',
                'flooding_api_notation': '026090',
            },
            {
                'label': 'Harpington Hill Farm',
                'hydrology_api_notation': 'bf61ce31-b20e-4593-85dc-a083133b12ce',
                'parameter': 'rainfall',
                'flooding_api_notation': '032822',
            },
            {
                'label': 'Knitlsey Mill',
                'hydrology_api_notation': '524a8fa0-d70b-4a0a-b178-ca765e73b8bc',
                'parameter': 'rainfall',
                'flooding_api_notation': '023839',
            },
            {
                'label': 'Fulwell',
                'hydrology_api_notation': '513abf6b-b269-4400-8828-7e833fd93eb8',
                'parameter': 'rainfall',
                'flooding_api_notation': '021028',
            },
            {
                'label': 'Stanhope',
                'hydrology_api_notation': 'b29c481a-5012-40f5-bb0c-f9370be34975',
                'parameter': 'level',
                'flooding_api_notation': '024003',
            },
            {
                'label': 'Durham New Elvet Bridge',
                'hydrology_api_notation': 'ba3f8598-e654-430d-9bb8-e1652e6ff93d',
                'parameter': 'level',
                'flooding_api_notation': '0240120',
            },
            {
                'label': 'Witton Park',
                'hydrology_api_notation': '05784319-693a-4d75-b29e-32f01a99ee4f',
                'parameter': 'level',
                'flooding_api_notation': '024008',
            },
            {
                'label': 'Chester Le Street',
                'hydrology_api_notation': 'e7d8bbb6-5bba-4057-9f49-a299482c3348',
                'parameter': 'level',
                'flooding_api_notation': '024009',
            },
            {
                'label': 'Sunderland Bridge',
                'hydrology_api_notation': 'ddedb4d9-b2be-47c1-998d-acbc0ffb124b',
                'parameter': 'level',
                'flooding_api_notation': '024001',
            },
        ]
    )
