from src.train import train, quiet_output
from src.config import Config
import argparse
import dataclasses

quiet_output()

parser = argparse.ArgumentParser()

for field in dataclasses.fields(Config):
    parser.add_argument(
        f"--{field.name}",
        type=field.type,
        default=field.default,
        help=f"Default: {field.default}",
    )

if __name__ == "__main__":
    args = parser.parse_args()
    train(Config(**vars(args)))
