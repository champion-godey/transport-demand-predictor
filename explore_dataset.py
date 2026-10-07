import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD DATASET
# ============================================================

file_name = "kumasi_transport_ml_ready.csv"

df = pd.read_csv(file_name)

print("=" * 60)
print("KUMASI PUBLIC TRANSPORT DATASET EXPLORATION")
print("=" * 60)

# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\nFIRST 5 RECORDS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ============================================================
# 3. DUPLICATES
# ============================================================

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# ============================================================
# 4. STATISTICAL SUMMARY
# ============================================================

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# ============================================================
# 5. LOCATIONS
# ============================================================

print("\nTRANSPORT LOCATIONS:")
print(df["location"].value_counts())

# ============================================================
# 6. TRANSPORT MODES
# ============================================================

print("\nTRANSPORT MODES:")
print(df["transport_mode"].value_counts())

# ============================================================
# 7. WEATHER
# ============================================================

print("\nWEATHER CONDITIONS:")
print(df["weather"].value_counts())

# ============================================================
# 8. DEMAND LEVEL
# ============================================================

print("\nDEMAND LEVELS:")
print(df["demand_level"].value_counts())

# ============================================================
# 9. AVERAGE DEMAND BY LOCATION
# ============================================================

print("\nAVERAGE DEMAND BY LOCATION:")

location_demand = (
    df.groupby("location")["boarding_count"]
    .mean()
    .sort_values(ascending=False)
)

print(location_demand)

# ============================================================
# 10. AVERAGE DEMAND BY TRANSPORT MODE
# ============================================================

print("\nAVERAGE DEMAND BY TRANSPORT MODE:")

mode_demand = (
    df.groupby("transport_mode")["boarding_count"]
    .mean()
    .sort_values(ascending=False)
)

print(mode_demand)

# ============================================================
# 11. AVERAGE DEMAND BY HOUR
# ============================================================

print("\nAVERAGE DEMAND BY HOUR:")

hour_demand = (
    df.groupby("hour")["boarding_count"]
    .mean()
)

print(hour_demand)

# ============================================================
# 12. VISUALIZE DEMAND BY HOUR
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    hour_demand.index,
    hour_demand.values,
    marker="o"
)

plt.title(
    "Average Public Transport Demand by Hour"
)

plt.xlabel("Hour of Day")

plt.ylabel("Average Passenger Boarding Count")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# 13. VISUALIZE DEMAND BY LOCATION
# ============================================================

plt.figure(figsize=(10, 6))

location_demand.plot(
    kind="bar"
)

plt.title(
    "Average Passenger Demand by Location"
)

plt.xlabel("Kumasi Location")

plt.ylabel("Average Boarding Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ============================================================
# 14. CORRELATION ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_columns.corr()

print("\nCORRELATION WITH BOARDING COUNT:")

print(
    correlation["boarding_count"]
    .sort_values(
        ascending=False
    )
)

print("\nDATA EXPLORATION COMPLETE.")