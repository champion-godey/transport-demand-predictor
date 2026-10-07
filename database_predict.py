import joblib
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta


# ============================================================
# 1. LOAD AI MODEL
# ============================================================

print("Loading AI model...")

model = joblib.load("transport_demand_model.pkl")

print("AI model loaded successfully!")


# ============================================================
# 2. CONNECT TO MYSQL
# ============================================================

try:

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kumasi_transport_ai"
    )

    cursor = connection.cursor(dictionary=True)

    print("MySQL connected successfully!")

except mysql.connector.Error as e:

    print("Database connection failed:")
    print(e)

    exit()


# ============================================================
# 3. GET USER INPUT
# ============================================================

print("\n========================================")
print("KUMASI TRANSPORT DEMAND PREDICTOR")
print("========================================")

location = input(
    "\nLocation "
    "(Kejetia, Adum, KNUST, Suame, Asokwa, "
    "Oforikrom, Bantama, Tafo, Ahodwo, Ejisu): "
)

transport_mode = input(
    "Transport mode (Trotro, Bus, Taxi): "
)

weather = input(
    "Weather (Clear, Cloudy, Rainy): "
)

date_input = input(
    "Date (YYYY-MM-DD): "
)

hour = int(
    input("Hour (0-23): ")
)


# ============================================================
# 4. CONVERT DATE
# ============================================================

selected_date = datetime.strptime(
    date_input,
    "%Y-%m-%d"
).date()


day_of_week = selected_date.weekday()

is_weekend = 1 if day_of_week >= 5 else 0

# For now, we manually use 0 for holiday.
# Later we can create a holiday table.
is_holiday = 0


# ============================================================
# 5. GET PREVIOUS HOUR DEMAND
# ============================================================

previous_hour = hour - 1

previous_date = selected_date

if previous_hour < 0:
    previous_hour = 23
    previous_date = selected_date - timedelta(days=1)


query = """
SELECT boarding_count
FROM transport_records
WHERE location = %s
AND transport_mode = %s
AND date = %s
AND hour = %s
LIMIT 1
"""

cursor.execute(
    query,
    (
        location,
        transport_mode,
        previous_date,
        previous_hour
    )
)

previous_hour_record = cursor.fetchone()


if previous_hour_record:

    demand_lag_1h = float(
        previous_hour_record["boarding_count"]
    )

else:

    demand_lag_1h = 0


# ============================================================
# 6. GET PREVIOUS DAY DEMAND
# ============================================================

previous_day = selected_date - timedelta(days=1)

cursor.execute(
    query,
    (
        location,
        transport_mode,
        previous_day,
        hour
    )
)

previous_day_record = cursor.fetchone()


if previous_day_record:

    demand_lag_24h = float(
        previous_day_record["boarding_count"]
    )

else:

    demand_lag_24h = 0


# ============================================================
# 7. GET PREVIOUS 3 HOURS
# ============================================================

three_hour_values = []

for i in range(1, 4):

    target_datetime = datetime.combine(
        selected_date,
        datetime.min.time()
    ) + timedelta(
        hours=hour - i
    )

    target_date = target_datetime.date()
    target_hour = target_datetime.hour

    cursor.execute(
        query,
        (
            location,
            transport_mode,
            target_date,
            target_hour
        )
    )

    record = cursor.fetchone()

    if record:

        three_hour_values.append(
            float(record["boarding_count"])
        )


if three_hour_values:

    demand_rolling_3h = sum(
        three_hour_values
    ) / len(three_hour_values)

else:

    demand_rolling_3h = 0


# ============================================================
# 8. GET PREVIOUS 24 HOURS
# ============================================================

twenty_four_hour_values = []

for i in range(1, 25):

    target_datetime = datetime.combine(
        selected_date,
        datetime.min.time()
    ) + timedelta(
        hours=hour - i
    )

    target_date = target_datetime.date()
    target_hour = target_datetime.hour

    cursor.execute(
        query,
        (
            location,
            transport_mode,
            target_date,
            target_hour
        )
    )

    record = cursor.fetchone()

    if record:

        twenty_four_hour_values.append(
            float(record["boarding_count"])
        )


if twenty_four_hour_values:

    demand_rolling_24h = sum(
        twenty_four_hour_values
    ) / len(twenty_four_hour_values)

else:

    demand_rolling_24h = 0


# ============================================================
# 9. CALCULATE SEASONAL FACTOR
# ============================================================

month = selected_date.month

if month in [12, 1, 2]:

    seasonal_factor = 1.10

elif month in [3, 4, 5]:

    seasonal_factor = 1.00

elif month in [6, 7, 8]:

    seasonal_factor = 0.95

else:

    seasonal_factor = 1.05


# ============================================================
# 10. GET TRAFFIC INDEX
# ============================================================

traffic_query = """
SELECT traffic_index
FROM transport_records
WHERE location = %s
AND date = %s
AND hour = %s
LIMIT 1
"""

cursor.execute(
    traffic_query,
    (
        location,
        selected_date,
        hour
    )
)

traffic_record = cursor.fetchone()


if traffic_record:

    traffic_index = float(
        traffic_record["traffic_index"]
    )

else:

    traffic_index = 50


# ============================================================
# 11. CREATE INPUT DATAFRAME
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
# 12. PREDICT
# ============================================================

prediction = model.predict(input_data)

predicted_demand = round(
    prediction[0],
    2
)


# ============================================================
# 13. DETERMINE DEMAND LEVEL
# ============================================================

if predicted_demand < 80:

    demand_level = "LOW"

elif predicted_demand < 160:

    demand_level = "MEDIUM"

else:

    demand_level = "HIGH"


# ============================================================
# 14. DISPLAY RESULTS
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
    f"Date: {selected_date}"
)

print(
    f"Hour: {hour}:00"
)

print(
    f"Weather: {weather}"
)

print(
    f"Previous Hour Demand: "
    f"{demand_lag_1h:.2f}"
)

print(
    f"Previous Day Demand: "
    f"{demand_lag_24h:.2f}"
)

print(
    f"3-Hour Average Demand: "
    f"{demand_rolling_3h:.2f}"
)

print(
    f"24-Hour Average Demand: "
    f"{demand_rolling_24h:.2f}"
)

print(
    f"Traffic Index: "
    f"{traffic_index:.2f}"
)

print("----------------------------------------")

print(
    f"Predicted Passenger Demand: "
    f"{predicted_demand}"
)

print(
    f"Demand Level: {demand_level}"
)

print("========================================")


# ============================================================
# 15. CLOSE DATABASE
# ============================================================

cursor.close()
connection.close()

print("\nDatabase connection closed.")
