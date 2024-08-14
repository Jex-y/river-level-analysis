import asyncio
import logging
from datetime import UTC, datetime
from typing import List, Union

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ValidationError,
    model_validator,
    Field,
    BeforeValidator,
)
from pymongo import InsertOne, UpdateOne
from typing_extensions import Optional, Self, Annotated
from enum import StrEnum
from bson.objectid import ObjectId

from .db import get_spills_collection


log = logging.getLogger("sewage_spills")

http_client = httpx.AsyncClient()

__all__ = ["collect_sewage_spills"]


class EventType(StrEnum):
    NO_RECENT_SPILL = "no recent spill"
    SPILL = "spill"
    MONITOR_OFFLINE = "monitor offline"


class EventMetadata(BaseModel):
    site_name: str
    site_id: str
    nearby: bool


class MongoDBDocument(BaseModel):
    id: Optional[Annotated[str, BeforeValidator(str)]] = Field(
        alias="_id", default=None
    )


class SpillEvent(MongoDBDocument):
    metadata: EventMetadata
    event_start: AwareDatetime
    event_end: AwareDatetime
    last_updated: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    event_type: EventType

    @model_validator(mode="after")
    def validate_event_times(self) -> Self:
        if self.event_start > self.event_end:
            raise ValidationError("Event start time must be before event end time")

    def __repr__(self) -> str:
        return (
            f"SpillEvent(site_name={self.metadata.site_name}, "
            f"event_type={self.event_type})"
        )

    def __str__(self) -> str:
        return self.__repr__()


WATERCOURSES = [
    "River Wear",
    "Pelaw Wood Beck",
    "Old Durham Beck",
    "Saltwell Gill",
    "River Browney",
    "Valley Burn",
    "Tributary of River Wear",
    "Holy Well Burn",
    "Bell Burn",
    "River Gaunless",
    "Tributary of River Wear (Culvert)",
    "Coal Burn",
    "Coundon Burn",
    "Beechburn Beck",
]

NEARBY_SITES = {
    "Millburngate SO",
    "Cathedral Banks SO",
    "Stockton Rd & Quarry Heads Ln SOs",
    "Barkers Haugh No 5 SO Prebends Bridge",
    "Church Street Du026",
    "Barclays A SO Elvet",
    "Cathedral Banks SO Barkers Haugh SO3 Court Lane West",
    # Looks like a typo in the data
    "Cathedral Banks SO Barkers Haugh SO3 Court Lane We",
    "Elvet Syphon SO",
    "Pelaw Wood Sewer, No 1 Baths Bridge SO",
    "Pelaw Wood St. Giles Close SO",
    "Laurel Avenue SO",
    "Shincliffe No 2 PS SO",
    "Shincliffe (A177) SO",
    "Jubilee Place SO",
    "Shincliffe SO Manhole No. 2",
    "Durham University STW SO Inlet, Durham University",
}


def maybe_parse_date(timestamp: int) -> datetime:
    if timestamp:
        return datetime.fromtimestamp(timestamp / 1000, tz=UTC)
    return None


def event_type_func(
    attributes: dict[str, Union[str, int]],
) -> EventType:
    start_time = maybe_parse_date(attributes["LAT_STIME"])
    end_time = maybe_parse_date(attributes["LAT_ETIME"])
    status = attributes["SP_STAT"]

    if "offline" in status:
        return EventType.MONITOR_OFFLINE

    if start_time and end_time:
        return EventType.SPILL

    return EventType.NO_RECENT_SPILL


async def load_query_data() -> List[SpillEvent]:
    base_url = httpx.URL(
        "https://services-eu1.arcgis.com/MSNNjkZ51iVh8yBj/arcgis/rest/services/Pledge2_view/FeatureServer/0/query"
    )

    watercourse_string = ", ".join([f"'{wc}'" for wc in WATERCOURSES])

    response = await http_client.get(
        base_url,
        params=dict(
            where=f"WC_NAME IN ({watercourse_string})",
            outFields="SITE_ID,SITE_NAME,SP_STAT,LAT_STIME,LAT_ETIME",
            returnDistinctValues=True,
            f="json",
        ),
    )

    try:
        result = [
            SpillEvent(
                metadata=EventMetadata(
                    site_name=x["attributes"]["SITE_NAME"],
                    site_id=x["attributes"]["SITE_ID"],
                    nearby=x["attributes"]["SITE_NAME"] in NEARBY_SITES,
                ),
                event_start=maybe_parse_date(x["attributes"]["LAT_STIME"])
                or datetime.now(tz=UTC),
                event_end=maybe_parse_date(x["attributes"]["LAT_ETIME"])
                or datetime.now(tz=UTC),
                event_type=event_type_func(x["attributes"]),
            )
            for x in response.json()["features"]
        ]
    except Exception as e:
        log.error(f"Error loading data from ArcGIS: {e}")
        raise e

    log.debug(f"Loaded {len(result)} records from ArcGIS")

    return result


async def load_db_data(site_ids: list[str]) -> List[SpillEvent]:
    return await asyncio.to_thread(
        lambda site_ids: [
            SpillEvent(**x)
            for x in get_spills_collection().aggregate(
                [
                    {"$match": {"metadata.site_id": {"$in": site_ids}}},
                    {
                        "$sort": {
                            "last_updated": -1,
                        }
                    },
                    {
                        "$group": {
                            "_id": "$metadata.site_id",
                            "event": {"$first": "$$ROOT"},
                        }
                    },
                    {
                        "$replaceRoot": {
                            "newRoot": "$event",
                        }
                    },
                ]
            )
        ],
        site_ids,
    )


async def bulk_write(operations: List):
    return await asyncio.to_thread(
        lambda operations: get_spills_collection().bulk_write(
            operations, ordered=False
        ),
        operations,
    )


def determine_operations(
    new_event: SpillEvent, prev_event: Optional[SpillEvent]
) -> List[Union[InsertOne, UpdateOne]]:
    # No previous event for this site, insert the new event
    if prev_event is None or (prev_event.event_type != new_event.event_type):
        reason = (
            "no previous event"
            if prev_event is None
            else f"different event type: {prev_event.event_type} != {new_event.event_type}"
        )
        log.debug(f"Inserting new event: {new_event} for reason: {reason}")
        return [InsertOne(new_event.model_dump())]

    # If an event is a spill, merge them if they overlap. Otherwise, insert the new event
    if new_event.event_type == prev_event.event_type == EventType.SPILL:
        if new_event.event_start > prev_event.event_end:
            log.debug(
                f"Inserting new event: {new_event} for reason: spills have no overlap"
            )
            return [InsertOne(new_event.model_dump())]

        updated_event_start = min(new_event.event_start, prev_event.event_start)
        updated_event_end = max(
            new_event.event_end or datetime.now(tz=UTC),
            prev_event.event_end,
        )

        if not (
            updated_event_start == prev_event.event_start
            and updated_event_end == prev_event.event_end
        ):
            log.debug(
                f"Updating event: {prev_event} to {updated_event_start} - {updated_event_end}"
            )
            return [
                UpdateOne(
                    {"_id": prev_event.id},
                    {
                        "$set": {
                            "event_start": updated_event_start,
                            "event_end": updated_event_end,
                        }
                    },
                ),
            ]

    # If they are the same event type (and not a spill), update the prev event end time as the event is ongoing
    if new_event.event_type == prev_event.event_type and (
        new_event.event_end > prev_event.event_end
    ):
        log.debug(
            f"Updating event: {prev_event} to end at {new_event.event_end} as it is ongoing"
        )

        return [
            UpdateOne(
                {"_id": prev_event.id},
                {
                    "$set": {
                        "event_end": new_event.event_end,
                    }
                },
            ),
        ]

    return []


def merge_events(
    query_events: List[SpillEvent], db_events: List[SpillEvent]
) -> List[Union[InsertOne, UpdateOne]]:
    db_events: dict[str, SpillEvent] = {
        db_event.metadata.site_id: db_event for db_event in db_events
    }

    return [
        op
        for query_event in query_events
        for op in determine_operations(
            query_event, db_events.get(query_event.metadata.site_id)
        )
    ]


async def collect_sewage_spills():
    log.info("Loading query data from ArcGIS")
    query_data = await load_query_data()

    log.info("Loading db data from MongoDB")
    site_ids = list({x.metadata.site_id for x in query_data})
    db_data = await load_db_data(site_ids)

    log.info("Merging data")
    operations = merge_events(query_data, db_data)

    num_inserted = sum(1 for op in operations if isinstance(op, InsertOne))
    num_updated = sum(1 for op in operations if isinstance(op, UpdateOne))
    log.info(f"Inserting {num_inserted} new records and updating {num_updated} records")

    await bulk_write(operations)

    log.info("Done")
