# 🚢 Titanic Survival Prediction - Stacking Ensemble Model

This project predicts passenger survival on the **Titanic dataset** using a **Stacking Ensemble Model** that combines multiple powerful algorithms like **CatBoost**, **Gradient Boosting**, and **XGBoost** with a **Logistic Regression** meta-model.  
It demonstrates a complete end-to-end ML pipeline — from data preprocessing and feature engineering to model tuning and evaluation.
------------------------------------------------------------------------------------------------------------------------------------------------
## 🧠 Project Overview

- **Goal:** Predict whether a passenger survived the Titanic disaster.
- **Machine Learning Task:** Binary Classification
- **Dataset Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
------------------------------------------------------------------------------------------------------------------------------------------------
# 📈 Model Performance

| Metric | Score |
|--------|-------:|
| **Training Accuracy** | 0.914 |
| **Testing Accuracy** | 0.859 |
| **Cross-Validation Score** | 0.828 |
| **ROC-AUC Score** | 0.887 |
| **F1-Score (Survived)** | 0.82 |

✅ The model generalizes well and achieves an excellent balance between precision and recall.
------------------------------------------------------------------------------------------------------------------------------------------------
## 🧮 Confusion Matrix
| Actual / Predicted | **No (0)** | **Yes (1)** |
|--------------------|-------------|-------------|
| **No (0)** | 92 | 13 |
| **Yes (1)** | 13 | 60 |
------------------------------------------------------------------------------------------------------------------------------------------------🧩 Tech Stack

Python 🐍

Pandas, NumPy

Scikit-Learn

CatBoost, XGBoost, LightGBM

Seaborn & Matplotlib for visualization

SMOTE for class balancing
------------------------------------------------------------------------------------------------------------------------------------------------
pip install -r requirements.txt

Live demo :- https://titanicpredictionmodel-kv3vvnk6lx8jwnhrk6gahf.streamlit.app/










👨‍💻 Author

Satyam Kumar
🎓 2nd-Year AI & ML Engineering Student
🔗 Linkedln :- www.linkedin.com/in/satyam-kumar-558269328
 | GitHub :- https://github.com/Satyam300702
 
