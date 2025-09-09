# Lab 3: Federated Learning Attacks and Countermeasure Strategies

## Overview

In Lab 1, we implemented a complete **Federated Learning (FL)** system using the FedAvg algorithm.  
In Lab 2, we explored the effects of **data heterogeneity** and **client drift**, and implemented advanced optimization techniques like **FedProx** and **SCAFFOLD**.  

Lab 3 focuses on the **security challenges in FL**. While FL keeps data local for privacy, it remains vulnerable to **adversarial behaviors** by participating clients.  
This lab investigates **data poisoning** and **model poisoning attacks**, and implements **robust aggregation methods**—**FedMedian** and **Krum**—to defend against malicious updates.  
You will also analyze the trade-offs these defenses introduce under varying levels of data heterogeneity.

**Key objectives:**
1. Understand **data poisoning** and **model poisoning** attacks in Federated Learning, including their goals, mechanisms, and effects on global model performance.  
2. Implement malicious client behaviors performing either data or model poisoning within a standard FL pipeline.  
3. Evaluate the impact of different proportions of malicious clients (e.g., 0%, 25%, 50%) on model performance using the FedAvg aggregation scheme.  
4. Implement and test two robust aggregation methods—**FedMedian** and **Krum**—and compare their effectiveness in mitigating poisoning attacks.  
5. Investigate how these defense mechanisms behave under different levels of data heterogeneity (using varying Dirichlet α values) and analyze the trade-offs between robustness to attacks and tolerance to legitimate variations among clients.

---

## Skills & Tools

- **Python**  
- **PyTorch**  
- **Flower (FL Framework)**  
- **Federated Learning security concepts**: Data poisoning, model poisoning, robust aggregation (FedMedian, Krum), attack-defense trade-offs  

---

## How to Explore the Lab

All experiments are available in **notebooks**. Open the notebook(s) in this folder and run cells sequentially to:  
- Implement malicious client behaviors.  
- Simulate attacks under different client proportions.  
- Apply robust aggregation methods to mitigate attacks.  
- Analyze performance under varying levels of data heterogeneity.  

---

## Expected Outcomes

For a detailed analysis of the results, please refer to the included **PDF report**:  

`Lab3_results.pdf`