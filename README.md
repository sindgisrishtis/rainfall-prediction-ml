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

---

## 📊 Dataset

The project uses daily meteorological observations containing atmospheric, temperature, humidity, wind, radiation, soil-moisture, and precipitation variables.

### Dataset Information

| Property | Details |
|---|---|
| **Time Period** | 1995–2025 |
| **Temporal Resolution** | Daily |
| **Location** | Latitude 18.8209, Longitude 98.9899 |
| **Target Variable** | `PRECTOTCORR` |
| **Prediction Task** | Daily rainfall regression |

### Meteorological Variables

| Feature | Description |
|---|---|
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave downward irradiance |
| `ALLSKY_SFC_SW_DNI` | All-sky surface direct normal irradiance |
| `T2M` | Temperature at 2 meters |
| `T2MDEW` | Dew-point temperature |
| `T2MWET` | Wet-bulb related temperature variable |
| `T2M_MAX` | Maximum temperature |
| `T2M_MIN` | Minimum temperature |
| `PRECTOTCORR` | Corrected precipitation / rainfall |
| `RH2M` | Relative humidity |
| `QV2M` | Specific humidity |
| `WS2M` | Wind speed |
| `GWETTOP` | Top-layer soil wetness |

---

## 🧹 Data Cleaning & Preprocessing

The raw meteorological data was prepared before performing feature engineering and model training.

The preprocessing workflow included:

- Loading and structuring the raw meteorological dataset
- Handling the dataset's header and metadata structure
- Selecting relevant numerical variables
- Checking data types and dataset consistency
- Checking for missing values
- Sorting observations chronologically
- Preparing the rainfall target variable
- Creating the cleaned modeling dataset
- Removing observations affected by lag and rolling-window feature generation

The cleaned dataset is maintained at:

`data/cleaned_rainfall_dataset.csv`

The preprocessing stage ensures that the data used for model development is structured, chronological, and suitable for downstream feature engineering and regression modeling.

---

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the statistical and temporal characteristics of rainfall before model training.

The analysis focused on:

- Rainfall distribution
- Rainfall trends over time
- Monthly rainfall seasonality
- Variability in rainfall
- Relationships between meteorological variables
- Correlation patterns
- Distribution of important weather variables

### 🌦️ Rainfall Seasonality

Monthly rainfall distributions were analyzed to identify seasonal patterns and differences in rainfall behavior across the year.

This analysis helps establish the importance of temporal information when modeling rainfall and provides a foundation for the subsequent feature-engineering stage.

EDA visualizations and project screenshots are available in the `assets/` directory.

---
