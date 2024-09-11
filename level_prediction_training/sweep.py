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
        "context_length": {
            "distribution": "q_uniform",
            "min": 4,
            "max": 4 * 24 * 2,
            "q": 4,
        },
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "train_epochs": {"distribution": "q_uniform", "min": 5, "max": 50, "q": 5},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-1,
        },
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "min": 64,
            "max": 2048,
            "q": 64,
        },
        "mlp_hidden_size": {
            "distribution": "q_log_uniform_values",
            "min": 8,
            "max": 256,
            "q": 8,
        },
        "num_mlp_blocks": {"distribution": "q_uniform", "min": 1, "max": 4, "q": 1},
        "num_conv_blocks": {"distribution": "q_uniform", "min": 1, "max": 4, "q": 1},
        "conv_kernel_size": {"values": [3, 5, 7]},
        "conv_hidden_size": {
            "distribution": "q_log_uniform_values",
            "min": 8,
            "max": 256,
            "q": 8,
        },
        "skip_connection": {"values": [True, False]},
        "dropout": {"distribution": "uniform", "min": 0.01, "max": 0.5},
        "activation_function": {
            "values": ["relu", "gelu", "tanh", "elu", "swish"],
        },
        "mlp_norm": {"values": ["batch", "layer", "none"]},
        "conv_norm": {"values": ["batch", "layer", "none"]},
        "norm_before_activation": {"values": [True, False]},
        "quantile_preprocessing_n_quantiles": {
            "distribution": "q_uniform",
            "min": 4,
            "max": 512,
            "q": 4,
        },
        "rolling_windows": {
            "values": [
                [
                    7 * 4 * 24,
                ],
                [
                    30 * 4 * 24,
                ],
                [
                    7 * 4 * 24,
                    30 * 4 * 24,
                ],
            ]
        },
    },
}

sweep_id = (
    wandb.sweep(sweep_config, project="river-level-forecasting")
    if args.sweep_id is None
    else args.sweep_id
)

quiet_output()


if args.n_agents == 1:
    wandb.agent(sweep_id, function=train, project="river-level-forecasting")
else:
    import multiprocessing as mp

    mp.set_start_method("spawn")
    for _ in range(args.n_agents):
        mp.Process(target=wandb.agent, args=(sweep_id, train)).start()
