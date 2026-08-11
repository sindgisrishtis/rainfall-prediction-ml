# 🌧️ Rainfall Prediction Using Machine Learning

<p align="center">
  <strong>End-to-end machine learning system for daily rainfall prediction using meteorological data, temporal feature engineering, XGBoost, and Streamlit.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Regression-FF6600?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Live-success?style=flat-square" alt="Status">
</p>

---

## 🚀 Live Demo

🌧️ **[Open the Live Rainfall Prediction Application](https://rainfall-prediction-ml-ydhagf8htgkxur45jbdnwp.streamlit.app)**

💻 **[View the Source Code on GitHub](https://github.com/sindgisrishtis/rainfall-prediction-ml)**

---

## Overview

**Rainfall Prediction Using Machine Learning** is an end-to-end regression project that predicts daily rainfall from historical meteorological observations and recent rainfall patterns.

The project combines **data preprocessing, exploratory data analysis, temporal feature engineering, time-aware model evaluation, ensemble learning, error analysis, and model deployment** into a complete machine learning workflow.

Multiple regression algorithms were trained and evaluated, including:

- Decision Tree
- Random Forest
- Support Vector Regression (SVR)
- XGBoost

Among the evaluated models, **XGBoost achieved the best test-set performance with an R² score of 0.9275**, making it the selected model for the deployed prediction application.

The trained model is serialized using Joblib and integrated into a **Streamlit web application** for interactive rainfall prediction.

---

## 🎯 Key Highlights

- 📊 Daily meteorological rainfall prediction
- 🧹 Data cleaning and preprocessing
- 🔎 Exploratory Data Analysis (EDA)
- 🧠 Temporal and rainfall-history feature engineering
- ⏳ Time-aware train/test splitting
- 🤖 Comparison of four machine learning regression models
- 🏆 XGBoost selected as the final model
- 📈 R², RMSE, MAE, and MSE evaluation
- 🔍 Feature importance analysis
- 📉 Residual and prediction error analysis
- 💾 Serialized trained model using Joblib
- 🌐 Interactive Streamlit application
- 🚀 Live cloud deployment
- 📁 Structured, reproducible project repository

---

## 📌 Problem Statement

Rainfall is influenced by multiple atmospheric and environmental factors, making accurate rainfall prediction a challenging regression problem.

The objective of this project is to build a machine learning model capable of learning relationships between meteorological conditions and recent rainfall behavior in order to estimate daily rainfall.

The project specifically investigates whether combining **meteorological variables with lagged and rolling rainfall features** can improve predictive performance.

---

## 🎯 Project Objectives

1. Prepare and clean historical meteorological data.
2. Analyze rainfall distributions, trends, and seasonal behavior.
3. Engineer meaningful temporal and rainfall-history features.
4. Train multiple regression-based machine learning models.
5. Evaluate model performance using appropriate regression metrics.
6. Compare models using an unseen test set.
7. Analyze feature importance and prediction errors.
8. Select the strongest-performing model.
9. Serialize the final model for inference.
10. Build and deploy an interactive rainfall prediction application.
