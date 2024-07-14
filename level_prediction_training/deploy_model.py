import argparse
from pathlib import Path

from rich.console import Console
from src.firebase import bucket

parser = argparse.ArgumentParser()
parser.add_argument(
    'model_dir',
    type=Path,
    help='Path to the model files',
)


def main():
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    pytorch_model = model_dir / 'model_torchscript.pt'
    preprocessing_pipeline = model_dir / 'preprocessing.pickle'
    stations = model_dir / 'stations.csv'

    assert model_dir.exists(), f'{model_dir} does not exist'
    assert pytorch_model.exists(), f'{pytorch_model} does not exist in {model_dir}'
    assert (
        preprocessing_pipeline.exists()
    ), f'{preprocessing_pipeline} does not exist in {model_dir}'
    assert stations.exists(), f'{stations} does not exist in {model_dir}'

    # Upload to model directory in bucket

    with Console().status('Uploading model'):
        blob = bucket.blob('model/model_torchscript.pt')
        blob.upload_from_filename(pytorch_model)

    with Console().status('Uploading preprocessing pipeline'):
        blob = bucket.blob('model/preprocessing.pickle')
        blob.upload_from_filename(preprocessing_pipeline)

    with Console().status('Uploading stations'):
        blob = bucket.blob('model/stations.csv')
        blob.upload_from_filename(stations)


if __name__ == '__main__':
    main()
