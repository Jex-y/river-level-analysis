from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum
from typing import Optional


class ActivationFunction(StrEnum):
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    ELU = "elu"
    SWISH = "swish"


class Norm(StrEnum):
    BATCH = "batch"
    LAYER = "layer"
    NONE = "none"


@dataclass
class Config:
    "Configuration for training the level forecast model"

    seed: Optional[int] = None
    "Seed for reproducibility. By default, a random seed is used."

    lr: float = 0.001
    "Learning rate for the optimizer"

    train_epochs: int = 100
    "Number of epochs to train the model"

    batch_size: int = 1024
    "Batch size for training"

    train_split: float = 0.8
    "Fraction of data to use for training"

    thresholds: tuple[float, ...] = (0.675,)
    "Thresholds to predict over/under probability"

    rolling_windows: tuple[int, ...] = (7 * 4 * 24, 28 * 4 * 24)

    quantiles: tuple[float, ...] = (0.05, 0.1, 0.9, 0.95)

    target_col: str = "Durham New Elvet Bridge - level"

    context_length: int = 4 * 12
    prediction_length: int = 4 * 12

    activation_function: ActivationFunction = ActivationFunction.SWISH

    mlp_norm: Norm = Norm.BATCH
    num_mlp_blocks: int = 4
    mlp_hidden_size: int = 64

    num_conv_blocks: int = 3
    conv_kernel_size: int = 3
    conv_hidden_size: int = 32
    conv_norm: Norm = Norm.BATCH
    skip_connection: bool = True

    dropout: float = 0.25
    weight_decay: float = 0.01

    model_save_dir: Path = Path("./models")
    stations_filepath: Path = Path("./stations.json")

    dev_run: bool = False
