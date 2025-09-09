import flwr as fl
from flwr.common import (GetPropertiesIns, 
                         GetPropertiesRes, 
                         GetParametersIns, 
                         GetParametersRes, 
                         FitIns, 
                         FitRes, 
                         EvaluateIns, 
                         EvaluateRes, 
                         ndarrays_to_parameters, 
                         parameters_to_ndarrays)
import torch
from torch.utils.data import DataLoader


class CustomClient(fl.client.Client):
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, device: torch.device) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.model.to(self.device)

    def get_properties(self, instruction: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="OK"),
            properties={}
        )

    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        params = self.model.get_model_parameters()
        parameters = ndarrays_to_parameters(params)
        return GetParametersRes(
            parameters=parameters,
            status=fl.common.Status(code=fl.common.Code.OK, message="OK")
        )

    def fit(self, instruction: FitIns) -> FitRes:
        global_params = parameters_to_ndarrays(instruction.parameters)
        self.model.set_model_parameters(global_params)

        epochs = instruction.config.get("epochs", 1)
        lr = instruction.config.get("lr", 0.01)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        total_loss = 0.0
        total_accuracy = 0.0

        for _ in range(epochs):
            loss, accuracy = self.model.train_epoch(
                self.train_loader, self.criterion, self.optimizer, self.device
            )
            total_loss += loss
            total_accuracy += accuracy

        avg_loss = total_loss / epochs
        avg_accuracy = total_accuracy / epochs

        updated_params = ndarrays_to_parameters(self.model.get_model_parameters())
        num_examples = len(self.train_loader.dataset)

        metrics = {"loss": avg_loss, "accuracy": avg_accuracy}

        return FitRes(
            parameters=updated_params,
            num_examples=num_examples,
            metrics=metrics,
            status=fl.common.Status(code=fl.common.Code.OK, message="OK")
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        global_params = parameters_to_ndarrays(instruction.parameters)
        self.model.set_model_parameters(global_params)

        loss, accuracy = self.model.test_epoch(self.test_loader, self.criterion, self.device)
        num_examples = len(self.test_loader.dataset)
        metrics = {"accuracy": accuracy}

        return EvaluateRes(
            loss=loss,
            num_examples=num_examples,
            metrics=metrics,
            status=fl.common.Status(code=fl.common.Code.OK, message="OK")
        )
