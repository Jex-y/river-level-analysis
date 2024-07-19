from .sewage_spills import load_query_data, merge_data
from datetime import datetime, timedelta
from pymongo import InsertOne, UpdateOne


async def test_load_query_data():
    spill_data = await load_query_data()

    assert isinstance(spill_data, list)
    assert len(spill_data) > 0

    schema = {
        'site_id': lambda x: isinstance(x, str) and x != '',
        'site_name': lambda x: isinstance(x, str) and x != '',
        'status': lambda x: isinstance(x, str) and x != '',
        'last_spill_start': lambda x: isinstance(x, datetime) or x is None,
        'last_spill_end': lambda x: isinstance(x, datetime) or x is None,
    }

    for site in spill_data:
        assert isinstance(site, dict)
        assert set(site.keys()) == set(schema.keys())

        for key, value in site.items():
            assert schema[key](value), f'Invalid format for {key}: {value}'

        # Last spill start should be before last spill end
        if site['last_spill_start'] and site['last_spill_end']:
            assert (
                site['last_spill_start'] <= site['last_spill_end']
            ), f"Invalid spill dates for {site['site_name']}"


def test_merge_data():
    query_spill_data = [
        {
            'site_id': '1',
            'site_name': 'Site 1',
            'status': 'Active',
            'last_spill_start': datetime(2020, 1, 1),
            'last_spill_end': datetime(2020, 1, 3),
        },
        {
            'site_id': '2',
            'site_name': 'Site 2',
            'status': 'Active',
            'last_spill_start': datetime(2020, 1, 1),
            'last_spill_end': datetime(2020, 1, 2),
        },
    ]

    db_spill_data = [
        {
            'metadata': {'site_name': 'Site 1', 'site_id': '1', 'nearby': False},
            '_id': '1',
            'event_start': datetime(2020, 1, 1),
            'event_end': datetime(2020, 1, 2),
            'event_type': 'spill',
        }
    ]

    operations = merge_data(query_spill_data, db_spill_data)

    assert len(operations) == 2
    assert operations == [
        UpdateOne(
            {'_id': '1'},
            {
                '$set': {
                    'event_start': datetime(2020, 1, 1),
                    'event_end': datetime(2020, 1, 3),
                }
            },
        ),
        InsertOne(
            {
                'metadata': {
                    'site_name': 'Site 2',
                    'site_id': '2',
                    'nearby': False,
                },
                'event_start': datetime(2020, 1, 1),
                'event_end': datetime(2020, 1, 2),
                'event_type': 'spill',
            }
        ),
    ]


def test_merge_monitor_offline():
    now = datetime.now()
    query_spill_data = [
        {
            'site_id': '1',
            'site_name': 'Site 1',
            'status': 'Monitor offline',
            'last_spill_start': None,
            'last_spill_end': None,
        }
    ]

    db_spill_data = [
        {
            'metadata': {
                'site_name': 'Site 1',
                'site_id': '1',
                'status': 'Monitor offline',
            },
            '_id': '1',
            'event_start': now - timedelta(hours=1),
            'event_end': now - timedelta(hours=1),
            'event_type': 'monitor offline',
        }
    ]

    operations = merge_data(query_spill_data, db_spill_data)

    assert len(operations) == 1

    assert isinstance(operations[0], UpdateOne)
    assert operations[0]._filter == {'_id': '1'}
    update = operations[0]._doc
    assert '$set' in update

    # the event start should be close to now - 1 hr
    assert update['$set']['event_start'] - (now - timedelta(hours=1)) < timedelta(
        minutes=1
    )
    assert update['$set']['event_end'] - now < timedelta(minutes=1)
