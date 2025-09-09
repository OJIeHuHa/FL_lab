import argparse
from server import main_server

def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--rounds", type=int, default=30, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--alpha", type=float, default=1, help="Dirichlet alpha for data split")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per client per round")
    parser.add_argument("--output", type=str, default=".", help="Directory to save json files")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset", type=str, default="fashionmnist", help="Dataset")
    parser.add_argument("--strategy", type=str, default="fedavg", 
                       choices=["fedavg", "fedprox", "scaffold"], help="Strategy")
    parser.add_argument("--mu", type=float, default=-1, help="FedProx proximal term coefficient (mu)")
    
    args = parser.parse_args()

    main_server(
        num_rounds=args.rounds,
        num_clients=args.clients,
        alpha=args.alpha,
        epochs=args.epochs,
        output_dir=args.output,
        lr=args.lr,
        batch_size=args.batch,
        dataset_name=args.dataset,
        strategy_name=args.strategy,
        mu=args.mu,
    )

if __name__ == "__main__":
    main()
