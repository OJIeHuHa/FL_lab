import flwr as fl
from flwr.common import (
    GetPropertiesIns, 
    GetPropertiesRes,
    GetParametersIns, 
    GetParametersRes,
    FitIns, 
    FitRes, 
    EvaluateIns, 
    EvaluateRes,
    ndarrays_to_parameters, 
    parameters_to_ndarrays
)
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import base64


class CustomClient(fl.client.Client):
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, device: torch.device, cid: int) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.model.to(self.device)
        self.cid = cid

        self.local_control = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]

    def get_properties(self, instruction: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="OK"),
            properties={}
        )

    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        params = self.model.get_model_parameters()
        return GetParametersRes(
            parameters=ndarrays_to_parameters(params),
            status=fl.common.Status(code=fl.common.Code.OK, message="OK")
        )
    
    def _log(self, message: str):
        log_filename = f"client_log_{self.cid}.txt"
        with open(log_filename, "a") as f:
            f.write(message + "\n")


    def decode_control_variate(self, encoded_str: str):
        decoded_bytes = base64.b64decode(encoded_str.encode('utf-8'))
        np_array_list = pickle.loads(decoded_bytes)
        tensor_list = [torch.tensor(arr, device=self.device, dtype=torch.float32) for arr in np_array_list]
        return tensor_list

    def fit(self, instruction: FitIns) -> FitRes:
        print("===FIT===")
        global_params = parameters_to_ndarrays(instruction.parameters)
        self.model.set_model_parameters(global_params)

        epochs = instruction.config.get("epochs", 1)
        lr = instruction.config.get("lr", 0.01)
        mu = instruction.config.get("mu", 0.0)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        global_control = None
        if self.model.strategy == "scaffold":
            if "c" not in instruction.config or "c_k" not in instruction.config:
                raise ValueError("Missing control variates for SCAFFOLD strategy")
            else:
                global_control = self.decode_control_variate(instruction.config["c"])
                self.local_control = self.decode_control_variate(instruction.config["c_k"])

            initial_params = [p.clone().detach() for p in self.model.parameters()]
            
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        for _ in range(epochs):
            if self.model.strategy == "fedprox":
                loss, accuracy, _ = self.model.train_epoch(
                    self.train_loader, self.criterion, self.optimizer, self.device,
                    global_params=global_params, mu=mu, learning_rate=lr
                )
            elif self.model.strategy == "scaffold":
                loss, accuracy, batch_count = self.model.train_epoch(
                    self.train_loader, self.criterion, self.optimizer, self.device,
                    local_control=self.local_control, global_control=global_control, learning_rate=lr
                )
                total_batches += batch_count
            else:
                loss, accuracy, _ = self.model.train_epoch(
                    self.train_loader, self.criterion, self.optimizer, self.device,
                    learning_rate=lr
                )

            total_loss += loss
            total_accuracy += accuracy

        avg_loss = total_loss / epochs
        avg_accuracy = total_accuracy / epochs

        num_examples = len(self.train_loader.dataset)

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
        }
        
        if self.model.strategy == "scaffold" and global_control is not None:
            print(f"Total batches:{total_batches}")
            updated_params = [p.clone().detach() for p in self.model.parameters()]
            for i in range(len(self.local_control)):
                delta_w = initial_params[i] - updated_params[i]
                update_term = delta_w / (lr * total_batches)
                self.local_control[i] = self.local_control[i] - global_control[i] + update_term

            c_k_new_numpy_list = [ck.detach().cpu().numpy() for ck in self.local_control]
            metrics["c_k_new_pickled"] = pickle.dumps(c_k_new_numpy_list)

        return FitRes(
            parameters=ndarrays_to_parameters(self.model.get_model_parameters()),
            num_examples=num_examples,
            metrics=metrics,
            status=fl.common.Status(code=fl.common.Code.OK, message="OK")
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        print("===EVALUATE===")
        global_params = parameters_to_ndarrays(instruction.parameters)
        self.model.set_model_parameters(global_params)

        loss, accuracy = self.model.test_epoch(self.test_loader, self.criterion, self.device)
        num_examples = len(self.test_loader.dataset)

        return EvaluateRes(
            loss=loss,
            num_examples=num_examples,
            metrics={"accuracy": accuracy},
            status=fl.common.Status(code=fl.common.Code.OK, message="OK")
        )