import random
import os

import flwr as fl
from data_utils import generate_distributed_datasets

from strategy import FedAvgStrategy, FedMedianStrategy, KrumStrategy
from flwr.common import ndarrays_to_parameters
from model import CustomFashionModel, CustomCifarModel
from manager import CustomClientManager
import numpy as np
import json

np.random.seed(42)
random.seed(42)




def main_server(num_rounds: int, num_clients: int, alpha: float, epochs: int, output_dir: str, lr: float, batch_size: int, dataset_name: str,
                strat: str, attack_type:str, attackers:float):
    data_dir = f"./distributed_data_{dataset_name}"
    generate_distributed_datasets(k=num_clients, alpha=alpha, save_dir=data_dir, dataset_name=dataset_name)
    print(f"Generated datasets for {num_clients} clients with alpha={alpha}")

    client_manager = CustomClientManager()

    if(dataset_name == "fashionmnist"):
        model = CustomFashionModel()
    elif(dataset_name=="cifar10"):
        model=CustomCifarModel()
    else:
        print("Invalid dataset name")
        return
    initial_weights = model.get_model_parameters()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    if strat == "fedavg":
        strategy = FedAvgStrategy(
            initial_parameters=initial_parameters,
            epochs=epochs,
            lr=lr,
            num_clients_fit=num_clients,
            num_clients_evaluate=num_clients,
        )
    elif strat=="fedmedian":
        strategy = FedMedianStrategy(
            initial_parameters=initial_parameters,
            epochs=epochs,
            lr=lr,
            num_clients_fit=num_clients,
            num_clients_evaluate=num_clients,
        )
    elif strat=="krum":
        strategy = KrumStrategy(
            initial_parameters=initial_parameters,
            epochs=epochs,
            lr=lr,
            f = int(num_clients*attackers),
            num_clients_fit=num_clients,
            num_clients_evaluate=num_clients,
        )
    else:
        print("Invalid strategy name")
        return

    print("Waiting for clients to connect before starting training...")
    print(f"Starting server on 127.0.0.1:8080 for {num_rounds} rounds")

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=60),
        strategy=strategy,
        client_manager=client_manager,
    )

    save_data(history, output_dir, num_rounds, epochs, num_clients, batch_size, lr, alpha, strat, attack_type, attackers)




def save_data(history, output_dir, num_rounds, epochs, num_clients, batch_size, lr, alpha, strategy, attack_type, attackers):
    os.makedirs(output_dir, exist_ok=True)

    history_data = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed,
    }
    with open(os.path.join(output_dir, "fl_history.json"), "w") as f:
        json.dump(history_data, f, indent=4)

    print("Saved training history to fl_history.json")

    plot1_path = os.path.join(output_dir, "plot1.json")
    plot1_data = []
    if os.path.exists(plot1_path):
        with open(plot1_path, "r") as f:
            plot1_data = json.load(f)

    plot1_entry = {
        "client": num_clients,
        "alpha": alpha,
        "strategy": strategy,
        "type": attack_type,
        "attackers": attackers,
        "loss_fit": history.metrics_distributed_fit.get("loss", []),
        "accuracy_fit": history.metrics_distributed_fit.get("accuracy", []),
        "loss_eval": history.losses_distributed,
        "accuracy_eval": history.metrics_distributed.get("accuracy", []),
    }

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
        "alpha": alpha,
        "client": num_clients,
        "strategy": strategy,
        "type": attack_type,
        "attackers": attackers,
        "accuracy": accuracy,
        "loss": loss,
    }

    plot2_data.append(plot2_entry)
    with open(plot2_path, "w") as f:
        json.dump(plot2_data, f, indent=4)
    print("Appended metadata and results to plot2.json")