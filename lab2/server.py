import random
import os

import flwr as fl
from data_utils import generate_distributed_datasets

from strategy import FedAvgStrategy, FedProxStrategy, ScaffoldStrategy
from flwr.common import ndarrays_to_parameters
from model import CustomFashionModel, CustomCifarModel
from manager import CustomClientManager
import numpy as np
import json

np.random.seed(42)
random.seed(42)


def main_server(num_rounds: int, num_clients: int, alpha: float, epochs: int, output_dir: str, lr: float, batch_size: int, dataset_name: str, strategy_name: str, mu: float):
    data_dir = f"./distributed_data_{dataset_name}"
    generate_distributed_datasets(k=num_clients, alpha=alpha, save_dir=data_dir, dataset_name=dataset_name)
    print(f"Generated datasets for {num_clients} clients with alpha={alpha}")

    client_manager = CustomClientManager()

    if(dataset_name == "fashionmnist"):
        model = CustomFashionModel(strategy=strategy_name)
    elif(dataset_name=="cifar10"):
        model=CustomCifarModel(strategy=strategy_name)
    else:
        print("Invalid dataset name")
        return
    initial_weights = model.get_model_parameters()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    if strategy_name.lower() == "fedavg":
        strategy = FedAvgStrategy(
            initial_parameters=initial_parameters,
            epochs=epochs,
            lr=lr,
            num_clients_fit=num_clients,
            num_clients_evaluate=num_clients,
        )
    elif strategy_name.lower() == "fedprox":
        strategy = FedProxStrategy(
            initial_parameters=initial_parameters,
            epochs=epochs,
            lr=lr,
            num_clients_fit=num_clients,
            num_clients_evaluate=num_clients,
            mu= 0.1 if mu==-1 else mu,
        )
    elif strategy_name.lower() == "scaffold":
        strategy = ScaffoldStrategy(
            initial_parameters=initial_parameters,
            epochs=epochs,
            lr=lr,
            num_clients_fit=num_clients,
            num_clients_evaluate=num_clients,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    print("Waiting for clients to connect before starting training...")
    print(f"Starting server on 127.0.0.1:8080 for {num_rounds} rounds")

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=60),
        strategy=strategy,
        client_manager=client_manager,
    )

    save_data(history, output_dir, num_rounds, epochs, num_clients, batch_size, lr, alpha, strategy_name, mu)




def save_data(history, output_dir, num_rounds, epochs, num_clients, batch_size, lr, alpha, strategy_name, mu):
    os.makedirs(output_dir, exist_ok=True)

    history_data = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed,
        "strategy": strategy_name,
    }
    with open(os.path.join(output_dir, "fl_history.json"), "w") as f:
        json.dump(history_data, f, indent=4)

    plot1_path = os.path.join(output_dir, "plot1.json")
    plot1_data = []
    if os.path.exists(plot1_path):
        with open(plot1_path, "r") as f:
            plot1_data = json.load(f)

    plot1_entry = {
        "rounds": num_rounds,
        "epoch": epochs,
        "client": num_clients,
        "batch": batch_size,
        "lr": lr,
        "alpha": alpha,
        "strategy": strategy_name,
        "loss_fit": history.metrics_distributed_fit.get("loss", []),
        "accuracy_fit": history.metrics_distributed_fit.get("accuracy", []),
        "loss_eval": history.losses_distributed,
        "accuracy_eval": history.metrics_distributed.get("accuracy", []),
    }
    if mu != -1:
        plot1_entry['mu'] = mu

    plot1_data.append(plot1_entry)
    with open(plot1_path, "w") as f:
        json.dump(plot1_data, f, indent=4)
    print("Appended metadata and full eval loss list to plot1.json")

    plot2_path = os.path.join(output_dir, "plot2.json")
    plot2_data = []
    if os.path.exists(plot2_path):
        with open(plot2_path, "r") as f:
            plot2_data = json.load(f)

    last_round = num_rounds
    accuracy_list = history.metrics_distributed.get("accuracy", [])
    loss_list = history.metrics_distributed.get("loss",[])
    accuracy = next((acc for rnd, acc in accuracy_list if rnd == last_round), None)
    loss = next((loss for rnd, loss in loss_list if rnd == last_round), None)

    plot2_entry = {
        "rounds": num_rounds,
        "epoch": epochs,
        "client": num_clients,
        "batch": batch_size,
        "lr": lr,
        "alpha": alpha,
        "strategy": strategy_name,
        "accuracy": accuracy,
        "loss": loss,
    }
    if mu != -1:
        plot2_entry['mu'] = mu

    plot2_data.append(plot2_entry)
    with open(plot2_path, "w") as f:
        json.dump(plot2_data, f, indent=4)
    print("Appended metadata and results to plot2.json")