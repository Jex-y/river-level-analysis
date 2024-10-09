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

    lr: float = 0.005
    "Learning rate for the optimizer"

    train_epochs: int = 25
    "Number of epochs to train the model"

    batch_size: int = 1024
    "Batch size for training"

    train_split: float = 0.8
    "Fraction of data to use for training"

    thresholds: tuple[float, ...] = (0.675,)
    "Thresholds to predict over/under probability"

    rolling_windows: tuple[int, ...] = (7 * 4 * 24, 30 * 4 * 24)

    quantiles: tuple[float, ...] = (0.1, 0.9)

    threshold_loss_coefficient: float = 1.0
    nnl_loss_coefficient: float = 1.0

    target_col: str = "Durham New Elvet Bridge - level"

    context_length: int = 4 * 8
    prediction_length: int = 4 * 12

    activation_function: ActivationFunction = ActivationFunction.GELU

    mlp_norm: Norm = Norm.BATCH
    num_mlp_blocks: int = 2
    mlp_hidden_size: int = 32

    num_conv_blocks: int = 4
    conv_kernel_size: int = 3
    conv_hidden_size: int = 64
    conv_norm: Norm = Norm.BATCH
    skip_connection: bool = True

    dropout: float = 0.2
    weight_decay: float = 0.01

    model_save_dir: Path = Path("./models")
    stations_filepath: Path = Path("./stations.json")

    dev_run: bool = False
