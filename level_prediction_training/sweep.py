from src.train import train, quiet_output
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sweep-id", type=str)
parser.add_argument("--n-agents", type=int, default=1)
args = parser.parse_args()


sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_total_loss", "goal": "minimize"},
    "early_terminate": {"type": "hyperband", "eta": 2, "min_iter": 8},
    "parameters": {
        "train_epochs": {"value": 150},
        "lr": {"distribution": "log_uniform_values", "min": 1e-3, "max": 5e-2},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 5e-2,
        },
        "mlp_hidden_size": {
            "distribution": "q_log_uniform_values",
            "min": 8,
            "max": 64,
            "q": 8,
        },
        "num_mlp_blocks": {"values": [3, 4, 5]},
        "num_conv_blocks": {"values": [0, 1, 2, 3, 4]},
        "conv_kernel_size": {"values": [3, 5]},
        "conv_hidden_size": {
            "distribution": "q_log_uniform_values",
            "min": 8,
            "max": 256,
            "q": 8,
        },
        "skip_connection": {"values": [True, False]},
        "activation_function": {
            "values": ["relu", "gelu", "tanh", "elu", "swish"],
        },
        "mlp_norm": {"values": ["batch", "layer"]},
        "conv_norm": {"values": ["batch", "layer"]},
        "norm_before_activation": {"values": [True, False]},
    },
}

sweep_id = (
    wandb.sweep(sweep_config, project="river-level-forecasting")
    if args.sweep_id is None
    else args.sweep_id
)

quiet_output()


wandb.agent(sweep_id, function=train, project="river-level-forecasting")
