import urllib3
import json
from typing import List, Dict
from datetime import datetime, timedelta
from pymongo import MongoClient, InsertOne, UpdateOne
from collections import defaultdict


nearby_sites = {
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
        return datetime.fromtimestamp(timestamp / 1000)
    return None


def load_query_data() -> List[Dict]:
    base_url = "https://services-eu1.arcgis.com/MSNNjkZ51iVh8yBj/arcgis/rest/services/Pledge2_view/FeatureServer/0/query"

    watercourses = [
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

    watercourse_string = ", ".join([f"'{wc}'" for wc in watercourses])

    response = urllib3.PoolManager().request(
        "GET",
        base_url,
        fields=dict(
            where=f"WC_NAME IN ({watercourse_string})",
            outFields="SITE_ID,SITE_NAME,SP_STAT,LAT_STIME,LAT_ETIME",
            returnDistinctValues=True,
            f="json",
        ),
    )
    json_data = json.loads(response.data.decode("utf-8"))
    return [
        {
            "site_id": x["attributes"]["SITE_ID"],
            "site_name": x["attributes"]["SITE_NAME"],
            "status": x["attributes"]["SP_STAT"],
            "last_spill_start": maybe_parse_date(x["attributes"]["LAT_STIME"]),
            "last_spill_end": maybe_parse_date(x["attributes"]["LAT_ETIME"]),
        }
        for x in json_data["features"]
    ]


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
        db_events[db_event["metadata"]["site_id"]].append(db_event)

    for query_event in query_event_data:
        # Status includes string "offline"
        event_type = (
            "monitor offline" if "offline" in query_event["status"].lower() else "spill"
        )

        if event_type == "spill" and (
            query_event["last_spill_start"] is None
            or query_event["last_spill_end"] is None
        ):
            # No recent spill data reported
            continue

        most_recent_event = (
            max(
                db_events[query_event["site_id"]],
                key=lambda x: x["event_end"] if x["event_end"] else datetime.min,
            )
            if db_events[query_event["site_id"]]
            else None
        )

        create_new_record = True

        match (
            event_type,
            most_recent_event["event_type"] if most_recent_event else None,
        ):
            case ("monitor offline", "monitor offline"):
                # If the last event was within the last 24 hours, extend it.
                # Otherwise, create a new one

                if most_recent_event["event_end"] > datetime.now() - timedelta(days=1):
                    event_start = min(most_recent_event["event_start"], datetime.now())
                    event_end = max(most_recent_event["event_end"], datetime.now())
                    create_new_record = False
                else:
                    event_start = datetime.now()
                    event_end = datetime.now()

            case ("spill", "spill"):
                # Check if the events overlap
                if (
                    query_event["last_spill_start"] <= most_recent_event["event_end"]
                    or query_event["last_spill_end"] >= most_recent_event["event_start"]
                ):
                    event_start = min(
                        most_recent_event["event_start"],
                        query_event["last_spill_start"],
                    )
                    event_end = max(
                        most_recent_event["event_end"], query_event["last_spill_end"]
                    )
                    create_new_record = False
                else:
                    event_start = query_event["last_spill_start"]
                    event_end = query_event["last_spill_end"]

            case ("monitor offline", "spill") | ("monitor offline", None):
                # Has just gone offline
                event_start = datetime.now()
                event_end = datetime.now()

            case ("spill", "monitor offline") | ("spill", None):
                # Has just come back online
                event_start = query_event["last_spill_start"]
                event_end = query_event["last_spill_end"]

            case _:
                raise ValueError(
                    f"Invalid event type: ({event_type} or {most_recent_event['type']})"
                )

        assert event_start <= event_end
        assert event_start is not None
        assert event_end is not None

        if (
            not create_new_record
            and event_start == most_recent_event["event_start"]
            and event_end == most_recent_event["event_end"]
        ):
            continue

        operations.append(
            InsertOne(
                {
                    "metadata": {
                        "site_name": query_event["site_name"],
                        "site_id": query_event["site_id"],
                        "nearby": query_event["site_name"] in nearby_sites,
                    },
                    "event_start": event_start,
                    "event_end": event_end,
                    "event_type": event_type,
                }
            )
            if create_new_record
            else UpdateOne(
                {"_id": db_event["_id"]},
                {
                    "$set": {
                        "event_start": min(db_event["event_start"], event_start),
                        "event_end": max(db_event["event_end"], event_end),
                    }
                },
            )
        )

    return operations


def get_collection():
    uri = "mongodb+srv://riverdata.mtspjxg.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=RiverData"
    client = MongoClient(
        uri,
        tls=True,
        tlsCertificateKeyFile="D:/code/river-level-analysis/water_pollution/db_cert.pem",
    )
    db = client["riverdata"]

    collection = db["spill-data"]

    if collection is None:
        collection = db.create_collection(
            "spill-data",
            timeseries=dict(
                timeField="event_end",
                metaField="metadata",
                granularity="hours",
            ),
        )
    return collection


if __name__ == "__main__":
    query_spill_data = load_query_data()

    oldest_spill = min(
        x["last_spill_start"] for x in query_spill_data if x["last_spill_start"]
    )

    collection = get_collection()

    db_spill_data = list(
        collection.find(
            {
                "event_start": {
                    "$gte": min(
                        oldest_spill, 
                        datetime.now() - timedelta(days=1)
                    )
                }
            }
        )
    )

    operations = merge_data(query_spill_data, db_spill_data)

    if operations:
        collection.bulk_write(operations, ordered=False)
        num_inserted = sum(1 for op in operations if isinstance(op, InsertOne))
        num_updated = sum(1 for op in operations if isinstance(op, UpdateOne))
        print(f"Inserted {num_inserted} new records and updated {num_updated} records")
    else:
        print("No new data")
