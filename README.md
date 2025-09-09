# Federated Learning Labs

---

## Contents
- [Lab 1: Introduction to Flower Framework](lab1/README.md)
- [Lab 2: Data Heterogeneity and Client Drift](lab2/README.md)
- [Lab 3: Federated Learning Attacks and Counteract Schemes](lab3/README.md)

---

## Overview

Federated Learning (FL) enables training machine learning models across decentralized data sources.

**Objectives of the labs:**
- **Lab 1:** Understand FL workflow, implement basic FedAvg strategy, and analyze hyperparameter impact on performance.  
- **Lab 2:** Study the impact of data heterogeneity on client drift and implement advanced federation strategies (FedProx, SCAFFOLD) to mitigate it.  
- **Lab 3:** Explore different types of attacks (data and model poisoning) and apply specialized strategies (FedMedian, KRUM) to defend against them.  

---

## Skills & Tools

- **Languages:** Python  
- **Frameworks:** PyTorch, Flower  
- **Concepts:** Data partitioning, simulation of distributed datasets and clients  
- **FL Strategies:** FedAvg, FedProx, SCAFFOLD, FedMedian, KRUM

---

## Setup

1. **Clone the repo**
    ```bash
    git clone https://github.com/OJIeHuHa/FL_lab
    cd FL_Lab
    ```

2. **Create environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate      # Linux/macOS
    .venv\Scripts\activate         # Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run notebooks or scripts**  
   Each lab folder contains a `README.md` with instructions for running the lab and expected outputs.


## Notes on code

Due to tight deadlines, all three labs were implemented within approximately 5 days.
The focus was on delivering **working, functional code** rather than detailed commenting of the code. 