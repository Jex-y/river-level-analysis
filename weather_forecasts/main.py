import urllib3
import json
from typing import List, Dict
from datetime import datetime
from pymongo import MongoClient


def get_collection():
    uri = 'mongodb+srv://riverdata.mtspjxg.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=RiverData'
    client = MongoClient(
        uri,
        tls=True,
        tlsCertificateKeyFile='./db_cert.pem',
    )
    db = client['riverdata']
    collection = db['weather-forecasts']

    if collection is None:
        collection = db.create_collection(
            'weather-forecasts',
            timeseries=dict(
                timeField='timestamp',
                metaField='metadata',
                granularity='minutes',
            ),
        )
    return collection


def get_forecast() -> Dict:
    raise NotImplementedError
