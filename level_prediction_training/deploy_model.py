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

    files = {
        'model': model_dir / 'model_torchscript.pt',
        'preprocessing pipeline': model_dir / 'preprocessing.pickle',
        'inference config': model_dir / 'inference_config.json',
    }

    assert model_dir.exists(), f'{model_dir} does not exist'

    for name, file in files.items():
        assert file.exists(), f'{name} does not exist in {model_dir}'

    for name, file in files.items():
        with Console().status(f'Uploading {name}'):
            blob = bucket.blob(f'model/{file.name}')
            blob.upload_from_filename(file)


if __name__ == '__main__':
    main()
