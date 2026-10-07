import pandas as pd
import mysql.connector
from mysql.connector import Error


# ============================================================
# 1. DATABASE CONNECTION
# ============================================================

try:

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kumasi_transport_ai"
    )

    cursor = connection.cursor()

    print("Database connected successfully!")


except Error as e:

    print("Database connection failed:")
    print(e)

    exit()


# ============================================================
# 2. LOAD CSV
# ============================================================

csv_file = "kumasi_transport_ml_ready.csv"

print("\nLoading dataset...")

df = pd.read_csv(csv_file)

print("Dataset loaded successfully!")
print("Rows:", len(df))


# ============================================================
# 3. PREPARE SQL QUERY
# ============================================================

sql = """
INSERT INTO transport_records (
    date,
    hour,
    location,
    transport_mode,
    weather,
    day_of_week,
    is_weekend,
    is_holiday,
    seasonal_factor,
    traffic_index,
    active_vehicles,
    boarding_count,
    occupancy_rate,
    delay_minutes,
    demand_level,
    demand_lag_1h,
    demand_lag_24h,
    demand_rolling_3h,
    demand_rolling_24h,
    anomaly_flag
)
VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
"""


# ============================================================
# 4. CONVERT DATA
# ============================================================

data = []

for _, row in df.iterrows():

    data.append((
        row["date"],
        int(row["hour"]),
        row["location"],
        row["transport_mode"],
        row["weather"],
        int(row["day_of_week"]),
        int(row["is_weekend"]),
        int(row["is_holiday"]),
        float(row["seasonal_factor"]),
        float(row["traffic_index"]),
        int(row["active_vehicles"]),
        float(row["boarding_count"]),
        float(row["occupancy_rate"]),
        float(row["delay_minutes"]),
        row["demand_level"],
        float(row["demand_lag_1h"]),
        float(row["demand_lag_24h"]),
        float(row["demand_rolling_3h"]),
        float(row["demand_rolling_24h"]),
        int(row["anomaly_flag"])
    ))


# ============================================================
# 5. INSERT DATA IN BATCHES
# ============================================================

print("\nImporting records...")
print("Please wait...")

batch_size = 5000

total_records = len(data)

for start in range(0, total_records, batch_size):

    batch = data[start:start + batch_size]

    cursor.executemany(sql, batch)

    connection.commit()

    imported = min(
        start + batch_size,
        total_records
    )

    print(
        f"Imported {imported:,} / {total_records:,} records"
    )


# ============================================================
# 6. CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()

print("\n========================================")
print("IMPORT COMPLETED SUCCESSFULLY")
print("========================================")

print(
    f"{total_records:,} records imported into MySQL."
)