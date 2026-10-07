import pandas as pd
import numpy as np

# ============================================================
# KUMASI PUBLIC TRANSPORT DEMAND DATASET GENERATOR
# ============================================================
# This creates SYNTHETIC data for a Kumasi-focused AI project.
# It is inspired by variables commonly used in public transport
# demand forecasting datasets.
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

locations = [
    "Kejetia",
    "Adum",
    "KNUST",
    "Suame",
    "Asokwa",
    "Oforikrom",
    "Bantama",
    "Tafo",
    "Ahodwo",
    "Ejisu"
]

transport_modes = [
    "Trotro",
    "Bus",
    "Taxi"
]

weather_conditions = [
    "Clear",
    "Cloudy",
    "Rainy"
]

# Approximate baseline passenger demand for simulation.
# These are NOT real measurements.
location_demand = {
    "Kejetia": 180,
    "Adum": 160,
    "KNUST": 145,
    "Suame": 135,
    "Asokwa": 125,
    "Oforikrom": 115,
    "Bantama": 110,
    "Tafo": 100,
    "Ahodwo": 95,
    "Ejisu": 90
}

mode_factor = {
    "Trotro": 1.00,
    "Bus": 0.75,
    "Taxi": 0.45
}

vehicle_capacity = {
    "Trotro": 18,
    "Bus": 45,
    "Taxi": 4
}

weather_factor = {
    "Clear": 1.00,
    "Cloudy": 0.96,
    "Rainy": 0.85
}

# ------------------------------------------------------------
# 2. CREATE DATE/TIME RANGE
# ------------------------------------------------------------

dates = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="h"
)

data = []

# ------------------------------------------------------------
# 3. GENERATE TRANSPORT RECORDS
# ------------------------------------------------------------

for timestamp in dates:

    date = timestamp.date()
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek

    is_weekend = 1 if day_of_week >= 5 else 0

    # Simple simulated public holiday list.
    holidays = [
        "2025-01-01",
        "2025-03-06",
        "2025-05-01",
        "2025-12-25",
        "2025-12-26"
    ]

    is_holiday = 1 if str(date) in holidays else 0

    # Select weather.
    weather = np.random.choice(
        weather_conditions,
        p=[0.55, 0.30, 0.15]
    )

    # --------------------------------------------------------
    # PEAK PERIODS
    # --------------------------------------------------------

    if 6 <= hour <= 9:
        peak_factor = 1.80

    elif 16 <= hour <= 19:
        peak_factor = 1.70

    elif 10 <= hour <= 15:
        peak_factor = 1.20

    elif 20 <= hour <= 22:
        peak_factor = 0.70

    elif 5 <= hour <= 23:
        peak_factor = 0.45

    else:
        peak_factor = 0.20

    # Weekend effect.
    weekend_factor = 0.72 if is_weekend else 1.0

    # Holiday effect.
    holiday_factor = 0.60 if is_holiday else 1.0

    # Seasonal variation.
    seasonal_factor = (
        1
        + 0.10 * np.sin(
            2 * np.pi * timestamp.dayofyear / 365
        )
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    for location in locations:

        for mode in transport_modes:

            # ------------------------------------------------
            # TRAFFIC INDEX
            # ------------------------------------------------

            traffic_index = 25

            if 6 <= hour <= 9:
                traffic_index += 35

            if 16 <= hour <= 19:
                traffic_index += 35

            if location in [
                "Kejetia",
                "Adum",
                "Suame",
                "Asokwa"
            ]:
                traffic_index += 15

            if weather == "Rainy":
                traffic_index += 12

            traffic_index += np.random.normal(0, 7)

            traffic_index = np.clip(
                traffic_index,
                0,
                100
            )

            # ------------------------------------------------
            # PASSENGER DEMAND
            # ------------------------------------------------

            base = location_demand[location]

            expected_demand = (
                base
                * mode_factor[mode]
                * peak_factor
                * weekend_factor
                * holiday_factor
                * seasonal_factor
                * weather_factor[weather]
            )

            # Random variation.
            boarding_count = np.random.poisson(
                max(expected_demand, 2)
            )

            boarding_count = max(
                0,
                int(boarding_count)
            )

            # ------------------------------------------------
            # ACTIVE VEHICLES
            # ------------------------------------------------

            capacity = vehicle_capacity[mode]

            active_vehicles = max(
                1,
                int(
                    np.ceil(
                        boarding_count / capacity
                    )
                )
            )

            # ------------------------------------------------
            # OCCUPANCY RATE
            # ------------------------------------------------

            total_capacity = (
                active_vehicles * capacity
            )

            occupancy_rate = (
                boarding_count /
                total_capacity
            )

            occupancy_rate += np.random.normal(
                0,
                0.04
            )

            occupancy_rate = np.clip(
                occupancy_rate,
                0.10,
                1.00
            )

            # ------------------------------------------------
            # DELAY
            # ------------------------------------------------

            delay_minutes = (
                1.5
                + (traffic_index * 0.07)
            )

            if weather == "Rainy":
                delay_minutes += 5

            delay_minutes += np.random.normal(
                0,
                2
            )

            delay_minutes = max(
                0,
                round(delay_minutes, 1)
            )

            # ------------------------------------------------
            # DEMAND LEVEL
            # ------------------------------------------------

            if boarding_count < 50:
                demand_level = "Low"

            elif boarding_count < 120:
                demand_level = "Moderate"

            elif boarding_count < 220:
                demand_level = "High"

            else:
                demand_level = "Very High"

            # ------------------------------------------------
            # STORE RECORD
            # ------------------------------------------------

            data.append([
                date,
                hour,
                location,
                mode,
                weather,
                day_of_week,
                is_weekend,
                is_holiday,
                round(seasonal_factor, 3),
                round(traffic_index, 2),
                active_vehicles,
                boarding_count,
                round(occupancy_rate, 3),
                delay_minutes,
                demand_level
            ])

# ------------------------------------------------------------
# 4. CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(
    data,
    columns=[
        "date",
        "hour",
        "location",
        "transport_mode",
        "weather",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "seasonal_factor",
        "traffic_index",
        "active_vehicles",
        "boarding_count",
        "occupancy_rate",
        "delay_minutes",
        "demand_level"
    ]
)

# ------------------------------------------------------------
# 5. SORT DATA
# ------------------------------------------------------------

df["datetime"] = (
    pd.to_datetime(df["date"])
    + pd.to_timedelta(df["hour"], unit="h")
)

df = df.sort_values(
    ["location", "transport_mode", "datetime"]
)

# ------------------------------------------------------------
# 6. CREATE HISTORICAL DEMAND FEATURES
# ------------------------------------------------------------

group = df.groupby(
    ["location", "transport_mode"]
)["boarding_count"]

# Previous hour.
df["demand_lag_1h"] = group.shift(1)

# Previous 24 hours.
df["demand_lag_24h"] = group.shift(24)

# Previous 3-hour average.
df["demand_rolling_3h"] = (
    group
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .mean()
    )
)

# Previous 24-hour average.
df["demand_rolling_24h"] = (
    group
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            24,
            min_periods=1
        )
        .mean()
    )
)

# ------------------------------------------------------------
# 7. ANOMALY DETECTION LABEL
# ------------------------------------------------------------

rolling_average = (
    df["demand_rolling_24h"]
    .fillna(df["boarding_count"])
)

df["anomaly_flag"] = (
    df["boarding_count"]
    > rolling_average * 1.8
).astype(int)

# ------------------------------------------------------------
# 8. REMOVE TEMPORARY COLUMN
# ------------------------------------------------------------

df = df.drop(
    columns=["datetime"]
)

# ------------------------------------------------------------
# 9. SAVE COMPLETE DATASET
# ------------------------------------------------------------

df.to_csv(
    "kumasi_public_transport_demand.csv",
    index=False
)

# ------------------------------------------------------------
# 10. CREATE ML-READY DATASET
# ------------------------------------------------------------

ml_df = df.dropna(
    subset=[
        "demand_lag_1h",
        "demand_lag_24h",
        "demand_rolling_3h",
        "demand_rolling_24h"
    ]
)

ml_df.to_csv(
    "kumasi_transport_ml_ready.csv",
    index=False
)

# ------------------------------------------------------------
# 11. DISPLAY INFORMATION
# ------------------------------------------------------------

print("=" * 60)
print("KUMASI PUBLIC TRANSPORT DATASET CREATED")
print("=" * 60)

print()

print(
    "Total records:",
    len(df)
)

print(
    "ML-ready records:",
    len(ml_df)
)

print()

print(
    "Locations:",
    len(locations)
)

print(
    "Transport modes:",
    len(transport_modes)
)

print()

print(
    "Main AI target: boarding_count"
)

print(
    "Secondary targets:"
)

print(
    "- occupancy_rate"
)

print(
    "- delay_minutes"
)

print(
    "- anomaly_flag"
)

print()

print(
    "Files created:"
)

print(
    "1. kumasi_public_transport_demand.csv"
)

print(
    "2. kumasi_transport_ml_ready.csv"
)

print("=" * 60)