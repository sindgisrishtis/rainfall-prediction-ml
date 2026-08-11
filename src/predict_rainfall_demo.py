import pickle
import numpy as np
print("Rainfall Prediction using Machine Learning")
print("------------------------------------------")

# Load the saved pickle model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Rainfall Prediction Model Loaded Successfully")

print("\nEnter weather values for the day:\n")

RH2M = float(input("Enter RH2M: "))
GWETTOP = float(input("Enter GWETTOP: "))
T2MDEW = float(input("Enter T2MDEW: "))
T2M_MIN = float(input("Enter T2M_MIN: "))
WS2M = float(input("Enter WS2M: "))
RAIN_LAG1 = float(input("Enter RAIN_LAG1: "))
RAIN_LAG2 = float(input("Enter RAIN_LAG2: "))
RAIN_LAG3 = float(input("Enter RAIN_LAG3: "))
RAIN_ROLLING3 = float(input("Enter RAIN_ROLLING3: "))
RAIN_ROLLING7 = float(input("Enter RAIN_ROLLING7: "))
RAIN_CHANGE = float(input("Enter RAIN_CHANGE: "))
RAIN_INTENSITY = float(input("Enter RAIN_INTENSITY: "))
TEMP_DEW_DIFF = float(input("Enter TEMP_DEW_DIFF: "))

# Convert input to array
user_input = np.array([[

RH2M,
GWETTOP,
T2MDEW,
T2M_MIN,
WS2M,
RAIN_LAG1,
RAIN_LAG2,
RAIN_LAG3,
RAIN_ROLLING3,
RAIN_ROLLING7,
RAIN_CHANGE,
RAIN_INTENSITY,
TEMP_DEW_DIFF

]])

# Predict rainfall
prediction = model.predict(user_input)
prediction_clipped = max(0.0, prediction[0])   # ← ADD THIS
print("Predicted Rainfall:", round(prediction_clipped, 3), "mm")