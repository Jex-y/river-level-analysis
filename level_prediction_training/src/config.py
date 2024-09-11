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


class PreprocessingType(StrEnum):
    STANDARD = "standard"
    QUANTILE = "quantile"
    MINMAX = "minmax"
    NONE = "none"


@dataclass
class Config:
    "Configuration for training the level forecast model"

    seed: Optional[int] = None
    "Seed for reproducibility. By default, a random seed is used."

    lr: float = 0.00006
    "Learning rate for the optimizer"

    train_epochs: int = 40
    "Number of epochs to train the model"

    batch_size: int = 448
    "Batch size for training"

    train_split: float = 0.8
    "Fraction of data to use for training"

    quantiles: tuple[float] = (0.1, 0.9)
    "Quantiles to predict"

    thresholds: tuple[float] = (0.675,)
    "Thresholds to predict over/under probability"

    rolling_windows: tuple[int] = (7 * 4 * 24, 30 * 4 * 24)

    quantile_loss_coefficient: float = 1.0
    threshold_loss_coefficient: float = 1.0
    mae_loss_coefficient: float = 1.0

    target_col: str = "Durham New Elvet Bridge - level"

    context_length: int = 4 * 4
    prediction_length: int = 4 * 12

    activation_function: ActivationFunction = ActivationFunction.SWISH
    norm: Norm = "batch"
    norm_before_activation: bool = False

    num_blocks: int = 3
    dropout: float = 0.2
    weight_decay: float = 0.0085
    hidden_size: int = 136

    model_save_dir: Path = Path("./models")
    stations_filepath: Path = Path("./stations.json")

    dev_run: bool = False

    level_preprocessing: PreprocessingType = PreprocessingType.QUANTILE
    rainfall_preprocessing: PreprocessingType = PreprocessingType.QUANTILE
    y_preprocessing: PreprocessingType = PreprocessingType.QUANTILE

    quantile_preprocessing_n_quantiles: int = 100
    quantile_preprocessing_output_normal = False
