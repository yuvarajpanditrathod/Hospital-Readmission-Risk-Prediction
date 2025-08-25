
# 🏥 Hospital Readmission Prediction Using Ensemble Learning
## 🚀 Badges & Tools
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![POS Tagging](https://img.shields.io/badge/POS-Tagging-yellow.svg)](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
[![Adversarial Training](https://img.shields.io/badge/Adversarial-Training-critical.svg)](https://arxiv.org/abs/1412.6572)
[![Dataset](https://img.shields.io/badge/Dataset-CSV-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xD_twIV6z6kD_YH3LAM0j-4j7Q6I7mUf)

---

## 🚀 Run on Google Colab

🔗 [Open the Notebook in Google Colab](https://colab.research.google.com/drive/1B3MA-TlDmo2XkLJtL1ShfQjgBawmjxZg)

---

## 📌 Overview
This project presents a comprehensive machine learning approach to predict hospital readmission rates using ensemble learning techniques. The study leverages the **FY 2024 Hospital Readmissions Reduction Program dataset** to develop predictive models that can help healthcare institutions identify patients at high risk of readmission and implement preventive measures.

## 📑 Abstract
Hospital readmissions represent a significant challenge in healthcare, affecting patient outcomes and increasing healthcare costs. This project implements advanced machine learning algorithms including **Multi-Layer Perceptron (MLP)**, **XGBoost**, and **CatBoost regressors** combined in an **ensemble learning framework** to predict excess readmission ratios. The ensemble approach demonstrates superior performance compared to individual models, achieving improved accuracy and robustness in predicting hospital readmission patterns.

---



## 📊 Dataset
The analysis utilizes the **FY 2024 Hospital Readmissions Reduction Program Hospital dataset**, which contains comprehensive information about hospital readmission metrics across various healthcare facilities in the United States.

### 🔑 Key Features:
- 🏥 **Facility Information**: Hospital names, IDs, and geographic location (State)
- 📉 **Readmission Metrics**: Number of discharges, readmissions, excess readmission ratios
- 📊 **Predictive Measures**: Predicted and expected readmission rates
- ⏳ **Temporal Data**: Start and end dates for measurement periods
- 🩺 **Measure Categories**: Different types of medical conditions and procedures

---

## 🛠️ Methodology

### 🔄 Data Preprocessing
1. 🧹 **Data Cleaning**: Removal of irrelevant columns (Facility Name, ID, Footnotes)
2. 🩹 **Missing Value Treatment**: Median imputation for numerical features
3. 🔤 **Categorical Encoding**: Label encoding and one-hot encoding for categorical variables
4. ⚖️ **Feature Standardization**: StandardScaler applied for neural network models
5. ✂️ **Data Splitting**: 80-20 train-test split with stratified sampling

### 🤖 Machine Learning Models
- **MLP Regressor**: Deep learning-based regressor with ReLU activation, Adam optimizer, and early stopping  
- **XGBoost Regressor**: Gradient boosting with optimized depth and learning rates  
- **CatBoost Regressor**: Categorical feature-friendly boosting algorithm  
- **Ensemble Learning**: Voting Regressor combining predictions for improved accuracy  

### 📏 Performance Evaluation Metrics
- 📉 **RMSE**
- 📈 **R² Score**
- ⚖️ **MAE**
- 📊 **MAPE**

---

## 🏆 Results

| Model | Training Accuracy | Test Accuracy |
|-------|------------------|---------------|
| 🔹 MLP Regressor | 98.33% | 97.76% |
| 🔹 XGBoost | 72.69% | 73.05% |
| 🔹 CatBoost | 71.82% | 71.90% |
| 🌟 **Ensemble Model** | **75.20%** | **75.29%** |

✅ **Findings**:  
- MLP achieves the highest accuracy but risks overfitting.  
- Ensemble learning provides more stable and generalizable predictions.  
- XGBoost and CatBoost show competitive performance around ~72-73%.  

---

## 📖 Research Publication
📌 This work has been accepted and published in:  
**"Hospital Readmission Prediction Using Ensemble Learning Techniques"**  
📝 **INCOFT 2025 Conference, Pune, India** 🎉  

The paper provides:
- Literature review and theoretical framework  
- Comprehensive methodology and experimental design  
- Comparative analysis with existing approaches  
- Clinical implications and recommendations  

---

## 🏥 Clinical Significance
- 🧾 **Risk Stratification**: Early detection of high-risk patients  
- 👩‍⚕️ **Decision Support**: Aid for clinicians in treatment planning  
- 💰 **Cost Reduction**: Reduced expenses through preventive measures  
- 📊 **Regulatory Compliance**: Quality reporting support  

---

## 🔮 Future Enhancements
- 🔗 Integration of deep learning architectures  
- ⏱️ Time-series analysis of patient data  
- 🧠 Multi-modal learning (EHR, imaging, clinical notes)  
- 🌍 Federated learning for cross-institutional collaboration  

---

## ⚙️ Usage Instructions
1. Clone the repo  
2. Install dependencies  
3. Load dataset  
4. Run preprocessing & training scripts  
5. Evaluate & visualize results  

---

## 🔐 License & Ethics
- 📜 Licensed under **MIT License**  
- ✅ Dataset is de-identified & publicly available  
- ⚖️ Complies with ethical guidelines in medical AI  

---

## 🤝 Contact & Collaboration
For collaborations, clinical discussions, or technical queries, please reach out via the contact information provided in the paper.

---

**Keywords**: Hospital Readmission, Ensemble Learning, Machine Learning, Healthcare Analytics, Predictive Modeling, XGBoost, CatBoost, Neural Networks, Clinical Decision Support  
