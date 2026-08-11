# 🌧️ Rainfall Prediction Using Machine Learning

<p align="center">
  <strong>An end-to-end machine learning system for daily rainfall prediction using meteorological data, temporal feature engineering, ensemble learning, and an interactive Streamlit application.</strong>
</p>

<p align="center">
  <a href="https://rainfall-prediction-ml-ydhagf8htgkxur45jbdnwp.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/sindgisrishtis/rainfall-prediction-ml">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Regression-FF6600?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

---

## 🚀 Live Demo

🌧️ **[Launch the Rainfall Prediction Application](https://rainfall-prediction-ml-ydhagf8htgkxur45jbdnwp.streamlit.app)**

💻 **[View the Source Code on GitHub](https://github.com/sindgisrishtis/rainfall-prediction-ml)**

> An interactive web application for exploring rainfall patterns, comparing machine learning models, analyzing predictions, and generating rainfall estimates using the trained XGBoost model.

---

## 📌 Overview

**Rainfall Prediction Using Machine Learning** is an end-to-end regression project that predicts daily rainfall from historical meteorological observations and recent rainfall patterns.

The project follows a complete machine learning workflow, beginning with data cleaning and exploratory analysis, followed by temporal feature engineering, chronological train/test splitting, model development, evaluation, error analysis, model serialization, and deployment.

Four regression models were evaluated:

- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)
- XGBoost Regressor

Among the evaluated models, **XGBoost achieved the best test-set performance with an R² score of 0.9275**, and was selected as the final model for the deployed application.

The trained model is serialized using **Joblib** and integrated into a **Streamlit application** that provides an interactive interface for rainfall prediction.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Dataset](#-dataset)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Time-Aware Data Splitting](#-time-aware-data-splitting)
- [ML Architecture & Workflow](#-ml-architecture--workflow)
- [Models Evaluated](#-models-evaluated)
- [Model Comparison & Final Model](#-model-comparison--final-model)
- [Feature Importance](#-feature-importance)
- [Error & Residual Analysis](#-error--residual-analysis)
- [Overfitting & Generalization](#-overfitting--generalization)
- [Streamlit Application](#-streamlit-application)
- [Screenshots](#-screenshots)
- [Project Structure](#-project-structure)
- [Run Locally](#-run-locally)
- [Deployment](#-deployment)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## ✨ Key Highlights

- 📊 Daily rainfall prediction using historical meteorological observations
- 🧹 Structured data cleaning and preprocessing pipeline
- 📈 Exploratory analysis of rainfall trends and seasonality
- 🧠 Temporal and rainfall-history feature engineering
- ⏳ Chronological train/test splitting to preserve temporal order
- 🤖 Comparison of four machine learning regression models
- 🏆 XGBoost selected as the final model
- 📐 Evaluation using R², RMSE, MAE, and MSE
- 🔍 Feature importance analysis
- 📉 Actual-vs-predicted and residual error analysis
- ⚠️ Training-vs-test generalization analysis
- 💾 Trained model serialized using Joblib
- 🌐 Interactive Streamlit prediction interface
- 🚀 Deployed using Streamlit Community Cloud
- 📁 Organized repository structure for reproducibility
