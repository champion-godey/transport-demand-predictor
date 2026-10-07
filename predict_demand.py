import joblib
import pandas as pd


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("Loading AI model...")

model = joblib.load("transport_demand_model.pkl")

print("AI model loaded successfully!")


# ============================================================
# 2. GET USER INPUT
# ============================================================

print("\n========================================")
print("KUMASI PUBLIC TRANSPORT DEMAND PREDICTOR")
print("========================================")


# Location
location = input(
    "\nEnter location "
    "(Kejetia, Adum, KNUST, Suame, Asokwa, "
    "Oforikrom, Bantama, Tafo, Ahodwo, Ejisu): "
)


# Transport mode
transport_mode = input(
    "Enter transport mode (Trotro, Bus, Taxi): "
)


# Weather
weather = input(
    "Enter weather (Clear, Cloudy, Rainy): "
)


# Hour
hour = int(
    input(
        "Enter hour (0-23): "
    )
)


# Day of week
day_of_week = int(
    input(
        "Enter day of week "
        "(0=Monday, 1=Tuesday, ..., 6=Sunday): "
    )
)


# Weekend
is_weekend = int(
    input(
        "Is it a weekend? "
        "(0=No, 1=Yes): "
    )
)


# Holiday
is_holiday = int(
    input(
        "Is it a holiday? "
        "(0=No, 1=Yes): "
    )
)


# Seasonal factor
seasonal_factor = float(
    input(
        "Enter seasonal factor "
        "(example: 1.00): "
    )
)


# Traffic index
traffic_index = float(
    input(
        "Enter traffic index "
        "(0-100): "
    )
)


# Previous hour demand
demand_lag_1h = float(
    input(
        "Enter passenger demand from previous hour: "
    )
)


# Previous day demand
demand_lag_24h = float(
    input(
        "Enter passenger demand from previous day: "
    )
)


# 3-hour rolling demand
demand_rolling_3h = float(
    input(
        "Enter average passenger demand for previous 3 hours: "
    )
)


# 24-hour rolling demand
demand_rolling_24h = float(
    input(
        "Enter average passenger demand for previous 24 hours: "
    )
)


# ============================================================
# 3. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([
    {
        "hour": hour,
        "location": location,
        "transport_mode": transport_mode,
        "weather": weather,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "seasonal_factor": seasonal_factor,
        "traffic_index": traffic_index,
        "demand_lag_1h": demand_lag_1h,
        "demand_lag_24h": demand_lag_24h,
        "demand_rolling_3h": demand_rolling_3h,
        "demand_rolling_24h": demand_rolling_24h
    }
])


# ============================================================
# 4. MAKE PREDICTION
# ============================================================

prediction = model.predict(input_data)

predicted_demand = round(
    prediction[0],
    2
)


# ============================================================
# 5. DETERMINE DEMAND LEVEL
# ============================================================

if predicted_demand < 80:
    demand_level = "LOW"

elif predicted_demand < 160:
    demand_level = "MEDIUM"

else:
    demand_level = "HIGH"


# ============================================================
# 6. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("AI PREDICTION RESULT")
print("========================================")

print(
    f"Location: {location}"
)

print(
    f"Transport Mode: {transport_mode}"
)

print(
    f"Weather: {weather}"
)

print(
    f"Predicted Passenger Demand: "
    f"{predicted_demand}"
)

print(
    f"Demand Level: {demand_level}"
)

print("========================================")