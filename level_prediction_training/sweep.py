from src.train import train, quiet_output
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sweep-id", type=str)
args = parser.parse_args()


sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_total_loss", "goal": "minimize"},
    "early_terminate": {"type": "hyperband", "eta": 2, "min_iter": 8},
    "parameters": {
        "train_epochs": {"value": 25},
        "lr": {"values": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]},
        "mlp_hidden_size": {
            "values": [8, 12, 16, 20, 24, 32],
        },
        "num_mlp_blocks": {"values": [2, 3, 4, 5]},
        "num_conv_blocks": {"values": [2, 3, 4, 5]},
        "conv_kernel_size": {"values": [3, 5]},
        "conv_hidden_size": {
            "values": [8, 16, 24, 32, 48, 64, 80, 96, 112, 128],
        },
        "activation_function": {
            "values": ["relu", "gelu", "tanh", "elu", "swish"],
        },
        "conv_norm": {"values": ["batch", "layer"]},
        "x_preprocessing": {"values": ["quantile", "none"]}
    },
}

sweep_id = (
    wandb.sweep(sweep_config, project="river-level-forecasting")
    if args.sweep_id is None
    else args.sweep_id
)

quiet_output()

wandb.agent(sweep_id, function=train, project="river-level-forecasting")
