import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import FashionMNIST, CIFAR10
import torchvision.transforms as transforms
import shutil

np.random.seed(42)

DATASETS = {
    "fashionmnist": (FashionMNIST, (0.2860,), (0.3530,)),
    "cifar10": (CIFAR10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}

def generate_distributed_datasets(k: int, alpha: float, save_dir: str, dataset_name: str) -> None:
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    dataset_name = dataset_name.lower()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    DatasetClass, mean, std = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    full_dataset = DatasetClass(root="./data", train=True, download=True, transform=transform)
    targets = np.array(full_dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(k)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        proportions = np.random.dirichlet(alpha=[alpha]*k)
        samples_per_client = (proportions * len(class_indices)).astype(int)
        while samples_per_client.sum() < len(class_indices):
            samples_per_client[np.random.randint(0, k)] += 1
        while samples_per_client.sum() > len(class_indices):
            samples_per_client[np.random.randint(0, k)] -= 1

        start_idx = 0
        for i, num_samples in enumerate(samples_per_client):
            selected_indices = class_indices[start_idx:start_idx+num_samples]
            client_indices[i].extend(selected_indices.tolist())
            start_idx += num_samples

    print(f"Number of clients: {len(client_indices)}")
    for i, indices in enumerate(client_indices):
        print(f"Client {i} has {len(indices)} samples")

    for i, indices in enumerate(client_indices):
        client_subset = Subset(full_dataset, indices)
        torch.save(client_subset, os.path.join(save_dir, f"client_{i}.pt"))



def load_client_data(cid: int, data_dir: str, batch_size: int):
    client_dataset: Dataset = torch.load(os.path.join(data_dir, f"client_{cid}.pt"), weights_only=False)
    total_len = len(client_dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_dataset, test_dataset = random_split(client_dataset, [train_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader