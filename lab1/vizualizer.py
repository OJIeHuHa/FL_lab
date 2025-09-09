import json
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from typing import Any, Dict, List
import argparse

class ResultsVisualizer:
    def __init__(self, folder: str, param_index: int) -> None:
        self.folder = folder
        self.param_index = param_index
        self.plot1_data: List[Dict[str, Any]] = []
        self.plot2_data: List[Dict[str, Any]] = []

    def load_data(self, annotate_dataset: bool = False) -> None:
        plot1_path = os.path.join(self.folder, "plot1.json")
        plot2_path = os.path.join(self.folder, "plot2.json")

        if not os.path.exists(plot1_path):
            raise FileNotFoundError(f"{plot1_path} not found")
        if not os.path.exists(plot2_path):
            raise FileNotFoundError(f"{plot2_path} not found")

        with open(plot1_path, "r") as f1:
            self.plot1_data = json.load(f1)

        with open(plot2_path, "r") as f2:
            self.plot2_data = json.load(f2)

        if not self.plot1_data:
            raise ValueError("plot1.json is empty")
        if not self.plot2_data:
            raise ValueError("plot2.json is empty")

        if annotate_dataset:
            datasets = ["CIFAR-10", "FashionMNIST"]
            half_1 = len(self.plot1_data) // 2
            half_2 = len(self.plot2_data) // 2

            for i, entry in enumerate(self.plot1_data):
                entry["dataset"] = datasets[0] if i < half_1 else datasets[1]

            for i, entry in enumerate(self.plot2_data):
                entry["dataset"] = datasets[0] if i < half_2 else datasets[1]

    def plot_single_simulation(self) -> None:
        entry = self.plot1_data[0]

        def plot_metrics(metrics: List[str], title: str, filename: str):
            plt.figure(figsize=(10, 6))
            for metric in metrics:
                if metric not in entry:
                    continue
                rounds = [r[0] for r in entry[metric]]
                values = [r[1] for r in entry[metric]]
                plt.plot(rounds, values, label=metric)
            plt.title(title)
            plt.xlabel("Round")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            output_path = os.path.join(self.folder, filename)
            plt.savefig(output_path)
            plt.close()
            print(f"Saved {title} plot to {output_path}")

        plot_metrics(["loss_fit", "accuracy_fit"], "Fit Metrics", "plot1.png")
        plot_metrics(["loss_eval", "accuracy_eval"], "Eval Metrics", "plot2.png")

    def plot_multi_simulations(self) -> None:
        example_entry = self.plot1_data[0]
        
        keys = [k for k in example_entry.keys() if k not in ("loss_fit", "accuracy_fit", "loss_eval", "accuracy_eval")]
        try:
            param_key = keys[self.param_index]
        except IndexError:
            raise IndexError(f"param_index {self.param_index} out of range. Available keys: {keys}")

        metrics = ["loss_fit", "accuracy_fit", "loss_eval", "accuracy_eval"]
        titles = ["Loss (Train)", "Accuracy (Train)", "Loss (Eval)", "Accuracy (Eval)"]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            for entry in self.plot1_data:
                if param_key not in entry or metric not in entry:
                    continue
                rounds = [r[0] for r in entry[metric]]
                values = [r[1] for r in entry[metric]]
                label = f"{param_key}={entry[param_key]}"
                axs[i].plot(rounds, values, label=label)

            axs[i].set_title(f"{titles[i]} by round")
            axs[i].set_xlabel("Round")
            axs[i].set_ylabel(titles[i])
            axs[i].grid(True)
            axs[i].legend()

        plt.tight_layout()
        plot_path_1 = os.path.join(self.folder, "plot1.png")
        plt.savefig(plot_path_1)
        plt.close()
        print(f"Saved metric curve plots to {plot_path_1}")

        x_vals = []
        y_vals = []

        for entry in self.plot1_data:
            if param_key in entry and "accuracy_eval" in entry:
                accuracy_list = entry["accuracy_eval"]
                if accuracy_list:
                    final_accuracy = accuracy_list[-1][1]
                    x_vals.append(entry[param_key])
                    y_vals.append(final_accuracy)

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, marker='o')
        plt.title(f"Final Accuracy (Eval) vs {param_key}")
        plt.xlabel(param_key)
        plt.ylabel("Final Accuracy (Eval)")
        plt.grid(True)
        plt.tight_layout()
        plot_path_2 = os.path.join(self.folder, "plot2.png")
        plt.savefig(plot_path_2)
        plt.close()
        print(f"Saved final accuracy vs param plot to {plot_path_2}")

    def print_results_table(self) -> None:
        table = PrettyTable()

        if not self.plot2_data:
            print("No data in plot2.json to print.")
            return

        example = self.plot2_data[0]
        field_names = [k for k in example.keys() if k not in ("accuracy", "loss")]
        field_names += ["accuracy", "loss"]
        table.field_names = field_names

        for entry in self.plot2_data:
            row = [entry.get(f, "N/A") for f in field_names]
            table.add_row(row)

        print(table)

def main():
    parser = argparse.ArgumentParser(description="Visualize Flower federated learning results with hyperparameter selection.")
    parser.add_argument("--folder", type=str, default=".", help="Folder containing plot1.json and plot2.json")
    parser.add_argument("--param_index", type=int, required=True, help="Index of hyperparameter to visualize")
    parser.add_argument("--dataset", action="store_true", help="Flag to annotate entries with dataset name (CIFAR-10/FashionMNIST)")
    parser.add_argument("--print_table", action="store_true", help="Print results table from plot2.json")
    parser.add_argument("--plot_type", type=int, choices=[1, 2], default=2,
                        help="Plot type: 1 for single simulation, 2 for hyperparameter comparison (default)")

    args = parser.parse_args()

    visualizer = ResultsVisualizer(args.folder, args.param_index)
    visualizer.load_data(annotate_dataset=args.dataset)

    if args.print_table:
        visualizer.print_results_table()

    if args.plot_type == 1:
        visualizer.plot_single_simulation()
    else:
        visualizer.plot_multi_simulations()


if __name__ == "__main__":
    main()
