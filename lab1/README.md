# Lab 1: Introduction to Flower Framework

## Overview

The first practical session (TP1) introduces the implementation of a **Federated Learning (FL) system** using the **Flower framework**.  
In this lab, you will gain hands-on experience with **client-server interactions**, the **FedAvg training strategy**, and the impact of **different hyperparameters** on model convergence and performance.

**Key objectives:**
1. **Generating Distributed Data**: Create a simulated distributed dataset with different class distributions across clients.  
2. **Designing a Model for FL**: Implement a simple machine learning model to be used by clients in federated training.  
3. **Implementing a Federated Client**: Extend `flwr.client.Client` to define client-side operations.  
4. **Running Individual Clients**: Explore how a client interacts with the server using its assigned dataset and model.  
5. **Implementing the Server’s Client Manager**: Extend `flwr.server.ClientManager` to manage participating clients.  
6. **Implementing a Basic Federated Learning Strategy**: Extend `flwr.server.Strategy` to define a basic aggregation strategy (FedAvg).  
7. **Running the Server**: Observe how the federated learning server coordinates client training.  
8. **Running a Full FL Simulation**: Execute the full simulation that starts the server and deploys multiple clients for federated training.  
9. **Analyzing FL with Different Configurations**: Conduct multiple simulations with varying hyperparameters (e.g., number of clients, data heterogeneity) and analyze their effects on model performance and convergence.  

---

## Skills & Tools

- **Python**  
- **PyTorch**  
- **Flower (FL Framework)**  
- **Federated Learning concepts**: Client-server interactions, FedAvg, data heterogeneity, hyperparameter tuning  

---

## How to Explore the Lab

All experiments are available in **notebooks**.  

- Each experiment is separated by a **header** in the notebook.  
- To run a specific experiment, simply **run all code cells below the corresponding header**.  
- **All results and logs** are saved into a **folder corresponding to that experiment**.  
- The **last cell of each experiment block** contains **visualizations** of the results for easy analysis.  

This allows you to:  
- Generate distributed datasets.  
- Implement clients and the server.  
- Run individual clients or full FL simulations.  
- Analyze the effects of different hyperparameters on model performance and convergence.

---

## Expected Outcomes

For a detailed analysis of the results, please refer to the included **PDF report**:  

`Lab1.pdf`