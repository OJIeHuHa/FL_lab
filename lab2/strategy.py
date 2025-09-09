from typing import List, Tuple, Optional, Dict, Union
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np
import time
import pickle
import base64

class FedAvgStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        epochs: int = 1,
        lr: float = 0.01,
        num_clients_fit: int = 3,
        num_clients_evaluate: int = 3,
    ):
        self.initial_parameters = initial_parameters
        self.epochs = epochs
        self.lr = lr
        self.num_clients_fit = num_clients_fit
        self.num_clients_evaluate = num_clients_evaluate

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        wait_time = 0
        max_wait = 15

        while len(client_manager._clients) < self.num_clients_fit:
            if wait_time >= max_wait:
                print(f"Timeout waiting for {self.num_clients_fit} clients in round {server_round}")
                break
            print(f"Waiting for clients in round {server_round}: "
                  f"{len(client_manager._clients)}/{self.num_clients_fit} connected...")
            time.sleep(1)
            wait_time += 1


        sample_clients = client_manager.sample(num_clients=self.num_clients_fit)
        config = {"epochs": self.epochs, "lr": self.lr}
        fit_ins = [
            (
                client,
                FitIns(parameters=parameters, config=config),
            )
            for client in sample_clients
        ]
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = []
        num_examples_total = 0
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((weights, fit_res.num_examples))
            num_examples_total += fit_res.num_examples

        new_weights = []
        for layer_i in range(len(weights_results[0][0])):
            layer_sum = sum(
                weights[layer_i] * num_examples for weights, num_examples in weights_results
            )
            new_weights.append(layer_sum / num_examples_total)

        total_loss = 0.0
        total_metrics = {}
        for _, fit_res in results:
            loss = fit_res.metrics.get("loss") if fit_res.metrics else None
            total_loss += loss * fit_res.num_examples if loss is not None else 0

            if fit_res.metrics:
                for key, value in fit_res.metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value * fit_res.num_examples

        avg_loss = total_loss / num_examples_total if num_examples_total > 0 else None
        avg_metrics = {
            key: value / num_examples_total for key, value in total_metrics.items()
        }

        aggregated_metrics: Dict[str, Scalar] = {"loss": avg_loss}
        aggregated_metrics.update(avg_metrics)

        parameters_aggregated = ndarrays_to_parameters(new_weights)

        print(f"\n[Round {server_round}] Fit results from clients:")
        for client, fit_res in results:
            print(
                f" - Client {client.cid}: "
                f"Loss: {fit_res.metrics.get('loss', 'N/A'):.4f}, "
                f"Accuracy: {fit_res.metrics.get('accuracy', 'N/A'):.4f}, "
                f"Examples: {fit_res.num_examples}"
            )

        print(f"[Round {server_round}] Aggregated training metrics: {aggregated_metrics}\n")

        return parameters_aggregated, aggregated_metrics


    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

        sample_clients = client_manager.sample(num_clients=self.num_clients_evaluate)
        eval_ins = [
            (client, EvaluateIns(parameters=parameters, config={})) for client in sample_clients
        ]
        return eval_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}


        total_loss = 0.0
        total_metrics = {}
        total_examples = 0

        for _, eval_res in results:
            num_examples = eval_res.num_examples
            total_loss += eval_res.loss * num_examples if eval_res.loss else 0
            for key, value in (eval_res.metrics or {}).items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * num_examples
            total_examples += num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else None
        avg_metrics = {k: v / total_examples for k, v in total_metrics.items()}

        aggregated_metrics: Dict[str, Scalar] = {"loss": avg_loss}
        aggregated_metrics.update(avg_metrics)

        print(f"\n[Round {server_round}] Evaluation results from clients:")
        for client, eval_res in results:
            print(f" - Client {client.cid}: "
                  f"Loss: {eval_res.loss:.4f}, "
                  f"Accuracy: {eval_res.metrics.get('accuracy', 'N/A'):.4f}, "
                  f"Examples: {eval_res.num_examples}")

        print(f"[Round {server_round}] Aggregated evaluation metrics: {aggregated_metrics}, Aggregated loss: {avg_loss:.4f}\n")

        return avg_loss, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None


class FedProxStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        epochs: int = 1,
        lr: float = 0.01,
        mu: float = 0.1,
        num_clients_fit: int = 3,
        num_clients_evaluate: int = 3,
    ):
        self.initial_parameters = initial_parameters
        self.epochs = epochs
        self.lr = lr
        self.mu = mu
        self.num_clients_fit = num_clients_fit
        self.num_clients_evaluate = num_clients_evaluate

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        wait_time = 0
        max_wait = 60
        while len(client_manager._clients) < self.num_clients_fit:
            if wait_time >= max_wait:
                print(f"Timeout waiting for {self.num_clients_fit} clients in round {server_round}")
                break
            print(f"Waiting for clients in round {server_round}: "
                  f"{len(client_manager._clients)}/{self.num_clients_fit} connected...")
            time.sleep(1)
            wait_time += 1

        sample_clients = client_manager.sample(num_clients=self.num_clients_fit)
        config = {"epochs": self.epochs, "lr": self.lr, "mu": self.mu}
        fit_ins = [(client, FitIns(parameters=parameters, config=config)) for client in sample_clients]
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = []
        num_examples_total = 0
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((weights, fit_res.num_examples))
            num_examples_total += fit_res.num_examples

        new_weights = []
        for layer_i in range(len(weights_results[0][0])):
            layer_sum = sum(
                weights[layer_i] * num_examples for weights, num_examples in weights_results
            )
            new_weights.append(layer_sum / num_examples_total)

        total_loss = 0.0
        total_metrics = {}
        for _, fit_res in results:
            loss = fit_res.metrics.get("loss") if fit_res.metrics else None
            total_loss += loss * fit_res.num_examples if loss is not None else 0

            if fit_res.metrics:
                for key, value in fit_res.metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value * fit_res.num_examples

        avg_loss = total_loss / num_examples_total if num_examples_total > 0 else None
        avg_metrics = {key: value / num_examples_total for key, value in total_metrics.items()}

        aggregated_metrics: Dict[str, Scalar] = {"loss": avg_loss}
        aggregated_metrics.update(avg_metrics)

        parameters_aggregated = ndarrays_to_parameters(new_weights)

        print(f"\n[Round {server_round}] FedProx fit results from clients:")
        for client, fit_res in results:
            print(
                f" - Client {client.cid}: "
                f"Loss: {fit_res.metrics.get('loss', 'N/A'):.4f}, "
                f"Accuracy: {fit_res.metrics.get('accuracy', 'N/A'):.4f}, "
                f"Examples: {fit_res.num_examples}"
            )
        print(f"[Round {server_round}] Aggregated FedProx training metrics: {aggregated_metrics}\n")

        return parameters_aggregated, aggregated_metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        sample_clients = client_manager.sample(num_clients=self.num_clients_evaluate)
        eval_ins = [(client, EvaluateIns(parameters=parameters, config={})) for client in sample_clients]
        return eval_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_loss = 0.0
        total_metrics = {}
        total_examples = 0
        for _, eval_res in results:
            num_examples = eval_res.num_examples
            total_loss += eval_res.loss * num_examples if eval_res.loss else 0
            for key, value in (eval_res.metrics or {}).items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * num_examples
            total_examples += num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else None
        avg_metrics = {k: v / total_examples for k, v in total_metrics.items()}

        aggregated_metrics: Dict[str, Scalar] = {"loss": avg_loss}
        aggregated_metrics.update(avg_metrics)

        print(f"\n[Round {server_round}] FedProx evaluation results from clients:")
        for client, eval_res in results:
            print(f" - Client {client.cid}: "
                  f"Loss: {eval_res.loss:.4f}, "
                  f"Accuracy: {eval_res.metrics.get('accuracy', 'N/A'):.4f}, "
                  f"Examples: {eval_res.num_examples}")

        print(f"[Round {server_round}] Aggregated FedProx evaluation metrics: {aggregated_metrics}, Aggregated loss: {avg_loss:.4f}\n")

        return avg_loss, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
    



class ScaffoldStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        epochs: int = 1,
        lr: float = 0.01,
        num_clients_fit: int = 3,
        num_clients_evaluate: int = 3,
        
    ):
        self.initial_parameters = initial_parameters
        self.epochs = epochs
        self.lr = lr
        self.num_clients_fit = num_clients_fit
        self.num_clients_evaluate = num_clients_evaluate

        self.server_c: Optional[List[np.ndarray]] = None
        self.client_cs: Dict[str, List[np.ndarray]] = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        param_arrays = parameters_to_ndarrays(self.initial_parameters)
        self.server_c = [np.zeros_like(p) for p in param_arrays]
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        wait_time = 0
        max_wait = 15

        while len(client_manager._clients) < self.num_clients_fit:
            if wait_time >= max_wait:
                print(f"Timeout waiting for {self.num_clients_fit} clients in round {server_round}")
                break
            print(f"Waiting for clients in round {server_round}: {len(client_manager._clients)}/{self.num_clients_fit} connected...")
            time.sleep(1)
            wait_time += 1
        
        print(f"Num Clients:{self.num_clients_fit}")
        sample_clients = client_manager.sample(num_clients=self.num_clients_fit//2)
        for client in sample_clients:
            if client.cid not in self.client_cs:
                self.client_cs[client.cid] = [np.zeros_like(c) for c in self.server_c]
            
            client_c = self.client_cs[client.cid]

            server_c_pickled = pickle.dumps(self.server_c)
            server_c_encoded = base64.b64encode(server_c_pickled).decode('utf-8')

            client_c_pickled = pickle.dumps(client_c)
            client_c_encoded = base64.b64encode(client_c_pickled).decode('utf-8')

            config = {
                "epochs": self.epochs,
                "lr": self.lr,
                "strategy": "scaffold",
                "c": server_c_encoded,
                "c_k": client_c_encoded  
            }

        fit_ins = [(client, FitIns(parameters=parameters, config=config)) for client in sample_clients]

        return fit_ins


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = []
        num_examples_total = 0
        delta_c_sum = [np.zeros_like(c) for c in self.server_c]

        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((weights, fit_res.num_examples))
            num_examples_total += fit_res.num_examples

            if "c_k_new_pickled" in fit_res.metrics:

                pickled_c_k_new = fit_res.metrics["c_k_new_pickled"]
                try:
                    deserialized_new_c_k = pickle.loads(pickled_c_k_new)
                except Exception as e:
                    print(f"Server: Error deserializing c_k_new for client {client.cid}: {e}")
                    continue

                if not isinstance(deserialized_new_c_k, list) or \
                   not all(isinstance(arr, np.ndarray) for arr in deserialized_new_c_k):
                    print(f"Server: Deserialized c_k_new for client {client.cid} is not a list of numpy arrays.")
                    continue

                self.client_cs[client.cid] = deserialized_new_c_k
                
                for i in range(len(delta_c_sum)):
                    delta_c_sum[i] += (deserialized_new_c_k[i] - self.server_c[i])

        new_weights = []
        for layer_i in range(len(weights_results[0][0])):
            layer_sum = sum(weights[layer_i] * num_examples for weights, num_examples in weights_results)
            new_weights.append(layer_sum / num_examples_total)

        for i in range(len(self.server_c)):
            self.server_c[i] = self.server_c[i] + delta_c_sum[i] / self.num_clients_fit

        total_loss = sum(fit_res.metrics["loss"] * fit_res.num_examples for _, fit_res in results)
        total_accuracy = sum(fit_res.metrics["accuracy"] * fit_res.num_examples for _, fit_res in results)
        
        avg_loss = total_loss / num_examples_total
        avg_accuracy = total_accuracy / num_examples_total

        aggregated_metrics: Dict[str, Scalar] = {"loss": avg_loss, "accuracy": avg_accuracy}
        parameters_aggregated = ndarrays_to_parameters(new_weights)

        print(f"\n[Round {server_round}] SCAFFOLD training results:")
        for client, fit_res in results:
            print(f" - Client {client.cid}: Loss = {fit_res.metrics['loss']:.4f}, "
                  f"Accuracy = {fit_res.metrics['accuracy']:.4f}, Examples = {fit_res.num_examples}")
        print(f"[Round {server_round}] Aggregated training metrics: {aggregated_metrics}\n")

        return parameters_aggregated, aggregated_metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        sample_clients = client_manager.sample(num_clients=self.num_clients_evaluate)
        return [(client, EvaluateIns(parameters=parameters, config={})) for client in sample_clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_loss = 0.0
        total_metrics = {}
        total_examples = 0
        
        for _, eval_res in results:
            num_examples = eval_res.num_examples
            total_loss += eval_res.loss * num_examples if eval_res.loss else 0
            for key, value in (eval_res.metrics or {}).items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * num_examples
            total_examples += num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else None
        avg_metrics = {k: v / total_examples for k, v in total_metrics.items()}

        aggregated_metrics: Dict[str, Scalar] = {"loss": avg_loss}
        aggregated_metrics.update(avg_metrics)

        print(f"\n[Round {server_round}] SCAFFOLD evaluation results from clients:")
        for client, eval_res in results:
            print(f" - Client {client.cid}: Loss: {eval_res.loss:.4f}, "
                  f"Accuracy: {eval_res.metrics.get('accuracy', 'N/A'):.4f}, Examples: {eval_res.num_examples}")
        print(f"[Round {server_round}] Aggregated evaluation metrics: {aggregated_metrics}, "
              f"Aggregated loss: {avg_loss:.4f}\n")

        return avg_loss, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None