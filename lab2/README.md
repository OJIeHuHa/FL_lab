# Lab 2: Data Heterogeneity and Client Drift

## Overview

In Lab 1, we implemented a customizable **Federated Learning (FL)** system using the **FedAvg scheme** and explored the impact of hyperparameters on federated training.  
Lab 2 builds on this foundation to investigate **data heterogeneity** and its effect on **client drift**, where local updates diverge from the global objective due to non-IID data.  

Additionally, this lab introduces and implements **advanced FL algorithms**—**FedProx** and **SCAFFOLD**—designed to mitigate the negative impact of heterogeneous data on model convergence.

**Key objectives:**
1. Simulate multiple FedAvg training runs with varying levels of data heterogeneity.  
2. Develop a conceptual understanding of data heterogeneity and client drift, and their effects on model convergence and performance.  
3. Explore and implement two advanced FL algorithms: **FedProx** and **SCAFFOLD**.  
4. Conduct simulations using FedProx and SCAFFOLD under different levels of data heterogeneity.  
5. Evaluate and compare the performance of FedAvg, FedProx, and SCAFFOLD in terms of accuracy, loss, and convergence speed.  
6. Analyze the effectiveness of each scheme in mitigating client drift and stabilizing training in heterogeneous FL settings.  

---

## Skills & Tools

- **Python**  
- **PyTorch**  
- **Flower (FL Framework)**  
- **Federated Learning concepts**: Client drift, data heterogeneity, FedAvg, FedProx, SCAFFOLD, algorithmic comparison  

---

## How to Explore the Lab

All experiments are available in **notebooks**.  

- Each experiment is separated by a **header** in the notebook.  
- To run a specific experiment, simply **run all code cells below the corresponding header**.  
- **All results and logs** are saved into a **folder corresponding to that experiment**.  
- The **last cell of each experiment block** contains **visualizations** of the results for easy analysis.  

This allows you to:  
- Simulate federated training under different levels of data heterogeneity.  
- Implement and test **FedProx** and **SCAFFOLD** algorithms.  
- Compare the performance of **FedAvg**, **FedProx**, and **SCAFFOLD**.  
- Analyze training stability, convergence, and the impact of client drift.

---

## Expected Outcomes

For a detailed analysis of the results, please refer to the included **PDF report**:  

`Lab2_results.pdf`