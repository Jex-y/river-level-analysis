import polars as pl
from src.dataset import load_inference_data
from pprint import pprint
from hydra.core.config_store import ConfigStore
from src.config import Config
import hydra
from pathlib import Path
import torch
import pickle
from datetime import timedelta

cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)


@hydra.main(version_base=None, config_name='config', config_path='.')
def main(config: Config):
    config.model_save_dir = Path(config.model_save_dir)
    preprocessing_file_name = 'confused-lake-84_preprocessing.pickle'

    preprocessing = pickle.load(
        open(config.model_save_dir / preprocessing_file_name, 'rb')
    )

    X_preprocessing = preprocessing['X']
    y_preprocessing = preprocessing['y']

    stations = pl.read_csv('stations.csv')

    model_file_name = 'confused-lake-84_torchscript.pt'
    model = torch.jit.load(config.model_save_dir / model_file_name, map_location='cpu')

    df = load_inference_data(stations, sequence_length=config.context_length)
    X = df.drop('dateTime').to_pandas().astype('float32')
    X = torch.tensor(X_preprocessing.transform(X)).unsqueeze(0)

    y_pred = model(X).detach().numpy().reshape(-1, 1)
    y_pred = y_preprocessing.inverse_transform(y_pred).reshape(
        config.prediction_length, len(config.quantiles)
    )

    assert config.quantiles[1] == 0.5

    # Predictions are at 15 minute intervals, starting from the last time in the input data
    y_pred_datetime = [
        df['dateTime'].max() + timedelta(minutes=15 * (i + 1))
        for i in range(config.prediction_length)
    ]

    values = [
        {'timestamp': dateTime, 'value': value, 'type': 'observed'}
        for dateTime, value in df[['dateTime', config.target_col]].iter_rows()
    ] + [
        {
            'timestamp': dateTime,
            'value': values[1],
            'type': 'predicted',
            'ci': [values[0], values[2]],
        }
        for dateTime, values in zip(y_pred_datetime, y_pred)
    ]

    print(values)


if __name__ == '__main__':
    main()
