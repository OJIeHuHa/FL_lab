import torch
from torch import nn
from typing import List, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader

class CustomFashionModel(nn.Module):
    def __init__(self, strategy: str = "fedavg") -> None:
        super().__init__()
        self.strategy = strategy
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_params: List[np.ndarray] = None,
        mu: float = 0.0,
        local_control: Optional[List[torch.Tensor]] = None,
        global_control: Optional[List[torch.Tensor]] = None,
        learning_rate: float = 0.01,
    ) -> Tuple[float, float, Optional[int]]:
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        global_tensors = None
        if self.strategy == "fedprox" and global_params is not None:
            global_tensors = [torch.tensor(p, dtype=torch.float32, device=device) for p in global_params]

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            if self.strategy == "fedprox" and global_tensors is not None:
                prox_term = sum(torch.norm(param - global_param) ** 2
                                for param, global_param in zip(self.parameters(), global_tensors))
                loss += (mu / 2) * prox_term

            loss.backward()

            if self.strategy == "scaffold" and local_control is not None and global_control is not None:
                with torch.no_grad():
                    for param, c_k, c in zip(self.parameters(), local_control, global_control):
                        if param.grad is not None:
                            param.grad.add_(c - c_k)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            num_batches += 1

        loss = running_loss / total
        accuracy = correct / total


        return loss, accuracy, num_batches

    def test_epoch(self, test_loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        loss = running_loss / total
        accuracy = correct / total
        return loss, accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.load_state_dict(state_dict)



class CustomCifarModel(nn.Module):
    def __init__(self, strategy: str = "fedavg") -> None:
        super().__init__()
        self.strategy = strategy
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_params: List[np.ndarray] = None,
        mu: float = 0.0,
        local_control: Optional[List[torch.Tensor]] = None,
        global_control: Optional[List[torch.Tensor]] = None,
        learning_rate: float = 0.01,
    ) -> Tuple[float, float, Optional[int]]:
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        global_tensors = None
        if self.strategy == "fedprox" and global_params is not None:
            global_tensors = [torch.tensor(p, dtype=torch.float32, device=device) for p in global_params]

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            if self.strategy == "fedprox" and global_tensors is not None:
                prox_term = sum(torch.norm(param - global_param) ** 2
                                for param, global_param in zip(self.parameters(), global_tensors))
                loss += (mu / 2) * prox_term

            loss.backward()

            if self.strategy == "scaffold" and local_control is not None and global_control is not None:
                with torch.no_grad():
                    for param, c_k, c in zip(self.parameters(), local_control, global_control):
                        if param.grad is not None:
                            param.grad.add_(c - c_k)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            num_batches += 1

        loss = running_loss / total
        accuracy = correct / total


        return loss, accuracy, num_batches

    def test_epoch(self, test_loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        loss = running_loss / total
        accuracy = correct / total
        return loss, accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.load_state_dict(state_dict)
