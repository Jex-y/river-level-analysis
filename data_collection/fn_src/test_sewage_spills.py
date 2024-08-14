from .sewage_spills import (
    load_query_data,
    SpillEvent,
    EventMetadata,
    EventType,
    determine_operations,
)
from datetime import datetime
from pymongo import InsertOne, UpdateOne
from pytest import mark, raises
from datetime import UTC
from pydantic import ValidationError
from bson.objectid import ObjectId


async def test_load_query_data():
    spill_data = await load_query_data()

    assert isinstance(spill_data, list)
    assert len(spill_data) > 0

    assert all(isinstance(x, SpillEvent) for x in spill_data)


def test_spill_event_requires_timezone():
    with raises(ValidationError):
        SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1),
            event_end=datetime(2020, 1, 2),
            event_type=EventType.SPILL,
        )

    SpillEvent(
        metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
        event_start=datetime(2020, 1, 1, tzinfo=UTC),
        event_end=datetime(2020, 1, 2, tzinfo=UTC),
        event_type=EventType.SPILL,
    )


class TestDetermineOperations:
    @mark.parametrize("event_type", [*EventType])
    def test_new_event(self, event_type):
        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=event_type,
        )

        operations = determine_operations(
            new_event,
            None,
        )

        assert operations == [InsertOne(new_event.model_dump())]

    def test_non_overlapping_spill(self):
        prev_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 3, tzinfo=UTC),
            event_end=datetime(2020, 1, 4, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        operations = determine_operations(
            new_event,
            prev_event,
        )

        assert operations == [InsertOne(new_event.model_dump())]

    def test_overlapping_spill(self):
        prev_event = SpillEvent(
            _id=ObjectId(),
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 4, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        operations = determine_operations(
            new_event,
            prev_event,
        )

        assert operations == [
            UpdateOne(
                {"_id": prev_event.id},
                {
                    "$set": {
                        "event_start": datetime(2020, 1, 1, tzinfo=UTC),
                        "event_end": datetime(2020, 1, 4, tzinfo=UTC),
                    }
                },
            )
        ]

    def test_adjacent_spill(self):
        prev_event = SpillEvent(
            _id=ObjectId(),
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 2, tzinfo=UTC),
            event_end=datetime(2020, 1, 4, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        operations = determine_operations(
            new_event,
            prev_event,
        )

        assert operations == [
            UpdateOne(
                {"_id": prev_event.id},
                {
                    "$set": {
                        "event_start": datetime(2020, 1, 1, tzinfo=UTC),
                        "event_end": datetime(2020, 1, 4, tzinfo=UTC),
                    }
                },
            )
        ]

    def test_equal_spill(self):
        prev_event = SpillEvent(
            _id=ObjectId(),
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        operations = determine_operations(
            new_event,
            prev_event,
        )

        assert operations == []

    @mark.parametrize(
        "event_type", [EventType.MONITOR_OFFLINE, EventType.NO_RECENT_SPILL]
    )
    def test_sequential_non_spill_events(self, event_type):
        prev_event = SpillEvent(
            _id=ObjectId(),
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=event_type,
        )

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 2, tzinfo=UTC),
            event_end=datetime(2020, 1, 3, tzinfo=UTC),
            event_type=event_type,
        )

        operations = determine_operations(new_event, prev_event)

        assert operations == [
            UpdateOne(
                {"_id": prev_event.id},
                {
                    "$set": {
                        "event_end": datetime(2020, 1, 3, tzinfo=UTC),
                    }
                },
            )
        ]

    @mark.parametrize(
        "event_types",
        [
            (EventType.MONITOR_OFFLINE, EventType.NO_RECENT_SPILL),
            (EventType.MONITOR_OFFLINE, EventType.SPILL),
            (EventType.NO_RECENT_SPILL, EventType.MONITOR_OFFLINE),
            (EventType.NO_RECENT_SPILL, EventType.SPILL),
            (EventType.SPILL, EventType.MONITOR_OFFLINE),
            (EventType.SPILL, EventType.NO_RECENT_SPILL),
        ],
    )
    def test_event_changed(self, event_types):
        prev_event_type, new_event_type = event_types

        prev_event = SpillEvent(
            _id=ObjectId(),
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=prev_event_type,
        )

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 2, tzinfo=UTC),
            event_end=datetime(2020, 1, 3, tzinfo=UTC),
            event_type=new_event_type,
        )

        operations = determine_operations(new_event, prev_event)

        assert operations == [InsertOne(new_event.model_dump())]

    def test_broken_case(self):
        prev_event = SpillEvent(
            _id=ObjectId(),
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 2, tzinfo=UTC),
            event_end=datetime(2020, 1, 3, tzinfo=UTC),
            event_type=EventType.MONITOR_OFFLINE,
        )

        # New event occurs before the previous event, however is just being reported now

        new_event = SpillEvent(
            metadata=EventMetadata(site_id="1", site_name="Site 1", nearby=False),
            event_start=datetime(2020, 1, 1, tzinfo=UTC),
            event_end=datetime(2020, 1, 2, tzinfo=UTC),
            event_type=EventType.SPILL,
        )

        operations = determine_operations(new_event, prev_event)

        assert operations == [InsertOne(new_event.model_dump())]
