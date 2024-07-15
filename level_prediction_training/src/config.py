from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    seed: int | None
    context_length: int
    prediction_length: int
    train_epochs: int
    lr: float
    batch_size: int
    train_epochs: int
    quantiles: list[float]
    target_col: str
    train_split: float

    activation_function: str
    num_blocks: int
    dropout: float
    weight_decay: float
    ff_dim: int
    normalize_before: bool
    norm_type: str
    model_save_dir: Path

    stations_filepath: Path
