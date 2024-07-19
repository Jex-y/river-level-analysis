import httpx
from pymongo import InsertOne, UpdateOne
import asyncio
from datetime import datetime, timedelta, UTC
from typing import List, Dict
from collections import defaultdict
from .db import get_spills_collection
import logging

log = logging.getLogger('sewage_spills')

http_client = httpx.AsyncClient()

__all__ = ['collect_sewage_spills']


WATERCOURSES = [
    'River Wear',
    'Pelaw Wood Beck',
    'Old Durham Beck',
    'Saltwell Gill',
    'River Browney',
    'Valley Burn',
    'Tributary of River Wear',
    'Holy Well Burn',
    'Bell Burn',
    'River Gaunless',
    'Tributary of River Wear (Culvert)',
    'Coal Burn',
    'Coundon Burn',
    'Beechburn Beck',
]

NEARBY_SITES = {
    'Millburngate SO',
    'Cathedral Banks SO',
    'Stockton Rd & Quarry Heads Ln SOs',
    'Barkers Haugh No 5 SO Prebends Bridge',
    'Church Street Du026',
    'Barclays A SO Elvet',
    'Cathedral Banks SO Barkers Haugh SO3 Court Lane West',
    # Looks like a typo in the data
    'Cathedral Banks SO Barkers Haugh SO3 Court Lane We',
    'Elvet Syphon SO',
    'Pelaw Wood Sewer, No 1 Baths Bridge SO',
    'Pelaw Wood St. Giles Close SO',
    'Laurel Avenue SO',
    'Shincliffe No 2 PS SO',
    'Shincliffe (A177) SO',
    'Jubilee Place SO',
    'Shincliffe SO Manhole No. 2',
    'Durham University STW SO Inlet, Durham University',
}


def maybe_parse_date(timestamp: int) -> datetime:
    if timestamp:
        return datetime.fromtimestamp(timestamp / 1000, tz=UTC)
    return None


async def load_query_data():
    base_url = httpx.URL(
        'https://services-eu1.arcgis.com/MSNNjkZ51iVh8yBj/arcgis/rest/services/Pledge2_view/FeatureServer/0/query'
    )

    watercourse_string = ', '.join([f"'{wc}'" for wc in WATERCOURSES])

    response = await http_client.get(
        base_url,
        params=dict(
            where=f'WC_NAME IN ({watercourse_string})',
            outFields='SITE_ID,SITE_NAME,SP_STAT,LAT_STIME,LAT_ETIME',
            returnDistinctValues=True,
            f='json',
        ),
    )

    result = [
        {
            'site_id': x['attributes']['SITE_ID'],
            'site_name': x['attributes']['SITE_NAME'],
            'status': x['attributes']['SP_STAT'],
            'last_spill_start': maybe_parse_date(x['attributes']['LAT_STIME']),
            'last_spill_end': maybe_parse_date(x['attributes']['LAT_ETIME']),
        }
        for x in response.json()['features']
    ]

    log.debug(f'Loaded {len(result)} records from ArcGIS: {result}')

    return result


async def load_db_data(since: datetime):
    return await asyncio.to_thread(
        lambda since: list(
            get_spills_collection().find({'event_start': {'$gte': since}})
        ),
        since,
    )


async def bulk_write(operations: List):
    return await asyncio.to_thread(
        lambda operations: get_spills_collection().bulk_write(
            operations, ordered=False
        ),
        operations,
    )


def merge_data(query_event_data: List[Dict], db_event_data: List[Dict]) -> List:
    # DB data has the schema:
    # {
    #     "metadata":
    #     {
    #         "site_name": "str",
    #         "site_id": "str",
    #     },
    #     "_id": "str",
    #     "event_start": "datetime",
    #     "event_end": "datetime | None",
    #     "event_type": "str", ("spill" or "monitor offline")
    # }

    # If the query spill overlaps with a corresponding db spill, update the db spill
    # If it does not overlap, add the query spill as a new record

    operations = []

    db_events = defaultdict(list)

    for db_event in db_event_data:
        db_events[db_event['metadata']['site_id']].append(db_event)

    for query_event in query_event_data:
        # Status includes string "offline"
        event_type = (
            'monitor offline' if 'offline' in query_event['status'].lower() else 'spill'
        )

        if event_type == 'spill' and (
            query_event['last_spill_start'] is None
            or query_event['last_spill_end'] is None
        ):
            # No recent spill data reported
            log.debug(f"No recent spill data for site '{query_event['site_name']}'")
            continue

        most_recent_event = (
            max(
                db_events[query_event['site_id']],
                key=lambda x: x['event_end']
                if x['event_end']
                else datetime(1970, 1, 1, tzinfo=UTC),
            )
            if db_events[query_event['site_id']]
            else None
        )

        create_new_record = True

        match (
            event_type,
            most_recent_event['event_type'] if most_recent_event else None,
        ):
            case ('monitor offline', 'monitor offline'):
                # If the last event was within the last 24 hours, extend it.
                # Otherwise, create a new one

                if most_recent_event['event_end'] > datetime.now(tz=UTC) - timedelta(
                    days=1
                ):
                    event_start = min(
                        most_recent_event['event_start'], datetime.now(tz=UTC)
                    )
                    event_end = max(
                        most_recent_event['event_end'], datetime.now(tz=UTC)
                    )
                    create_new_record = False
                else:
                    event_start = datetime.now(tz=UTC)
                    event_end = datetime.now(tz=UTC)

            case ('spill', 'spill'):
                # Check if the events overlap
                if (
                    query_event['last_spill_start'] <= most_recent_event['event_end']
                    or query_event['last_spill_end'] >= most_recent_event['event_start']
                ):
                    event_start = min(
                        most_recent_event['event_start'],
                        query_event['last_spill_start'],
                    )
                    event_end = max(
                        most_recent_event['event_end'], query_event['last_spill_end']
                    )
                    create_new_record = False
                else:
                    event_start = query_event['last_spill_start']
                    event_end = query_event['last_spill_end']

            case ('monitor offline', 'spill') | ('monitor offline', None):
                # Has just gone offline
                log.debug(f"Site '{query_event['site_name']}' has gone offline")

                event_start = datetime.now(tz=UTC)
                event_end = datetime.now(tz=UTC)

            case ('spill', 'monitor offline') | ('spill', None):
                # Has just come back online
                log.debug(f"Site '{query_event['site_name']}' has come back online")

                event_start = query_event['last_spill_start']
                event_end = query_event['last_spill_end']

            case _:
                raise ValueError(
                    f"Invalid event type: ({event_type} or {most_recent_event['type']})"
                )

        assert event_start <= event_end
        assert event_start is not None
        assert event_end is not None

        if (
            not create_new_record
            and event_start == most_recent_event['event_start']
            and event_end == most_recent_event['event_end']
        ):
            continue

        operations.append(
            InsertOne(
                {
                    'metadata': {
                        'site_name': query_event['site_name'],
                        'site_id': query_event['site_id'],
                        'nearby': query_event['site_name'] in NEARBY_SITES,
                        'event_id': f"{query_event['site_id']}_{event_start.timestamp()}",
                    },
                    'event_start': event_start,
                    'event_end': event_end,
                    'event_type': event_type,
                }
            )
            if create_new_record
            else UpdateOne(
                {'_id': db_event['_id']},
                {
                    '$set': {
                        'event_start': min(db_event['event_start'], event_start),
                        'event_end': max(db_event['event_end'], event_end),
                    },
                },
            )
        )

    return operations


# async def generate_sewage_spills_report():
#     # report:
#     # - Spills in last 24 hours and whether they are nearby
#     # - Spills in last week and whether they are nearby
#     # - Nearby monitors that are currently offline

#     recent_events = await loop.run_in_executor(
#         None,
#         get_spills_collection().find(
#             {'event_end': {'$gte': datetime.now(tz=UTC) - timedelta(days=7)}}
#         ),
#     )

#     last_24_hours =


async def collect_sewage_spills():
    log.info('Loading query data from ArcGIS')
    query_data = await load_query_data()

    oldest_spill = min(
        x['last_spill_start'] for x in query_data if x['last_spill_start']
    )

    log.info('Loading db data from MongoDB')
    db_data = await load_db_data(
        min(oldest_spill, datetime.now(tz=UTC) - timedelta(days=1))
    )

    log.info('Merging data')
    operations = merge_data(query_data, db_data)

    num_inserted = sum(1 for op in operations if isinstance(op, InsertOne))
    num_updated = sum(1 for op in operations if isinstance(op, UpdateOne))
    log.info(f'Inserting {num_inserted} new records and updating {num_updated} records')

    await bulk_write(operations)

    # log.info('Generating report')

    # log.info('Uploading report to bucket')

    log.info('Done')
