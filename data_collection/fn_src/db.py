from datetime import UTC

from bson.codec_options import CodecOptions
from firebase_functions.params import StringParam
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

__all__ = ["get_spills_collection", "get_weather_forecasts_collection"]

db_uri = StringParam("DB_URI")

client = None
db = None


def get_client():
    global client
    if client is None:
        client = MongoClient(db_uri.value, server_api=ServerApi("1"))
    return client


def get_db():
    global db
    if db is None:
        db = get_client()["riverdata"]
    return db


def get_spills_collection():
    return get_db()["spills"].with_options(
        codec_options=CodecOptions(tz_aware=True, tzinfo=UTC)
    )


def get_weather_forecasts_collection():
    return get_db()["weather-forecasts"].with_options(
        codec_options=CodecOptions(tz_aware=True, tzinfo=UTC)
    )
