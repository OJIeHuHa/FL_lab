import argparse
import torch
import flwr as fl
from client import CustomClient
from model import CustomFashionModel, CustomCifarModel
from data_utils import load_client_data

def run_client():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument('--dataset',type=str, default="fashionmnist", help="Dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = f"./distributed_data_{args.dataset}"

    train_loader, val_loader = load_client_data(args.cid, data_dir, args.batch)
    if(args.dataset == "fashionmnist"):
        model = CustomFashionModel()
    elif(args.dataset=="cifar10"):
        model = CustomCifarModel()
    else:
        print("Invalid dataset name")
        return
    
    client = CustomClient(model, train_loader, val_loader, device)

    fl.client.start_client(
        server_address="localhost:8080",
        client=client
    )

if __name__ == "__main__":
    run_client()
