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

    DatasetClass, mean, std = DATASETS[dataset_name.lower()]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    full_dataset = DatasetClass(root="./data", train=True, download=True, transform=transform)
    targets = np.array(full_dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(k)]

    # Split each class among clients
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        proportions = np.random.dirichlet(alpha=[alpha]*k)
        samples_per_client = (proportions * len(class_indices)).astype(int)

        # Adjust to match total number of samples
        while samples_per_client.sum() < len(class_indices):
            samples_per_client[np.random.randint(0, k)] += 1
        while samples_per_client.sum() > len(class_indices):
            samples_per_client[np.random.randint(0, k)] -= 1

        start_idx = 0
        for i, num_samples in enumerate(samples_per_client):
            client_indices[i].extend(class_indices[start_idx:start_idx+num_samples].tolist())
            start_idx += num_samples

    # Save datasets
    for i, indices in enumerate(client_indices):
        torch.save(Subset(full_dataset, indices), os.path.join(save_dir, f"client_{i}.pt"))


def load_client_data(cid: int, data_dir: str, batch_size: int):
    # Load client dataset
    client_dataset: Dataset = torch.load(os.path.join(data_dir, f"client_{cid}.pt"), weights_only=False)

    # Split into train/test
    total_len = len(client_dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_dataset, test_dataset = random_split(client_dataset, [train_len, test_len])

    # Return DataLoaders
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False)