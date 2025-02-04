# FUNCXAI-11

This repository provides the implementation of a **benchmark framework for Explainable Artificial Intelligence (XAI) methods**. The framework evaluates XAI techniques based on **11 key properties (F1–F11)** and applies them to **LIME, KernelSHAP, and TreeSHAP** using the **Pima Indians dataset** and a **Random Forest classifier**. The purpose of this benchmark is to help researchers and practitioners assess the strengths and weaknesses of XAI methods in a structured manner.  

## Repository Structure  

- `metrics.py` – Contains all functions defining the **numerical formulation of the evaluation metrics** for the framework.  
- `framework_example.ipynb` – A Jupyter Notebook that guides the reader through the evaluation process, applying the benchmark to **LIME, KernelSHAP, and TreeSHAP**.  
- `generated/` – Stores the results of the benchmark evaluations:  
  - CSV files containing scores for all **FUNCXAI-11 properties**.  
  - `ALL.csv` – Aggregated scores for all evaluated methods.  
  - Spider plots used in **Section 4** of the paper to visualize the properties of the XAI methods.  


## How to Use  

To follow the evaluation step-by-step, open **`framework_example.ipynb`** and execute the cells sequentially. This notebook provides:  
- A demonstration of how to **compute and compare the 11 properties** for different XAI methods.  
- The **application of the framework** using a Random Forest classifier as the black-box model trained on the Pima Indians dataset.  
- The **generation of results and visualization**.  

Ensure all dependencies are installed before running the scripts.  


## Dependencies 

 - Python (3.10.10)
 - Python packages: *numpy*, *pandas*, *shap*, *lime*, *sklearn*, *matplotlib*, *scipy*, *dice_ml*, *time* (optional: *seaborn*, *os*, *warnings*)


## Published Work

For a detailed explanation of the framework, methodology, and results, please refer to the associated paper:

 - *TO BE ADDED*

---

For questions or contributions, feel free to open an issue in this repository. 🚀
