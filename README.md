# CM2604-Cousework_2425604_Tharuka
Machine Learning Coursework focused on predicting Customer Churn using Decision Trees and Neural Networks.

# Telco Customer Churn Prediction: A Comparative Analysis of Supervised Learning Architectures

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 1. Project Overview
Customer attrition, or "churn," is a critical metric in the telecommunications sector. This project implements a robust machine learning pipeline to predict whether a customer will leave the service provider based on demographic, service, and financial attributes.

The study conducts a comparative analysis between two distinct supervised learning paradigms:
* **Decision Tree Classifier:** A symbolic, interpretability-focused model.
* **Artificial Neural Network (ANN):** A connectionist, performance-focused deep learning model.

## 2. Key Features
* **Corpus Preparation:** Comprehensive data cleaning, feature engineering (Tenure Grouping), One-Hot Encoding, and Min-Max Scaling.
* **Imbalance Handling:** Implementation of **SMOTE (Synthetic Minority Over-sampling Technique)** to rectify the 73:27 class imbalance.
* **Exploratory Data Analysis (EDA):** Statistical investigation using Targeted Heatmaps and KDE plots to identify key churn drivers.
* **Hyperparameter Optimization:**
    * **Decision Tree:** `GridSearchCV` for depth and splitting criteria.
    * **Neural Network:** Custom iterative grid search for Batch Size and Epochs.
* **Evaluation:** Comparative assessment using AUC-ROC, Recall, F1-Score, and Confusion Matrices.

## 3. Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Decision Tree, Metrics, Preprocessing)
* **Deep Learning:** TensorFlow/Keras (Sequential ANN)
* **Data Ingestion:** KaggleHub

## 4. Methodology
The project follows a rigorous experimental design:
1.  **Data Ingestion:** Automated download from Kaggle using `kagglehub`.
2.  **Preprocessing:** Handling missing values in `TotalCharges` and removing non-predictive identifiers.
3.  **Feature Engineering:** Creating `Tenure_Group` to capture lifecycle stages.
4.  **Balancing:** Application of SMOTE to synthesize minority class instances.
5.  **Modeling:** Training a Decision Tree (max_depth=5 initially) and a Multi-Layer Perceptron (16-8-8-1 architecture with Dropout).
6.  **Optimization:** Tuning hyperparameters to maximize generalization.

## 5. Results Summary
The experimental results demonstrate a clear trade-off between interpretability and predictive power.

| Metric | Decision Tree (Optimized) | Neural Network (Optimized) |
| :--- | :--- | :--- |
| **Accuracy** | ~79% | **~81%** |
| **AUC-ROC** | ~0.83 | **~0.86** |
| **Interpretability** | **High (White Box)** | Low (Black Box) |

* **Key Insight:** The Neural Network outperformed the Decision Tree in discriminatory power (AUC), likely due to its ability to capture non-linear interactions between features like `MonthlyCharges` and `Tenure`. However, the Decision Tree provided actionable "White Box" rules regarding Contract types.

## 6. Installation & Usage
To replicate this analysis:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TharukaGamage19/CM2604-Cousework_2425604_Tharuka]
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn kagglehub
    ```
3.  **Run the Notebook:**
    Open `CM2606_Coursework_Tharuka.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially. The dataset will be downloaded automatically via the API.

## 7. Future Enhancements
* **Ensemble Methods:** Implementation of XGBoost or LightGBM to bridge the gap between interpretability and performance.
* **MLOps:** Development of a pipeline to retrain the model quarterly to address concept drift.
* **Unstructured Data:** Integration of customer sentiment data from support logs.

## 8. References
* BlastChar. (2018). *Telco Customer Churn*. [Kaggle].
* Chawla, N.V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*.
* UNESCO. (2021). *Recommendation on the Ethics of Artificial Intelligence*.

---
*Author: Tharuka Gamage*
