import ujson as json
import boto3
import sklearn
import pickle
import os
from ...shared.hydrology_api import HydrologyApi, Measure, process_hydrology_data


def lambda_handler(event, context):
    """Update the data bucket and make predictions

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    CACHED_DATA_BUCKET = os.environ['CACHED_DATA_BUCKET']
