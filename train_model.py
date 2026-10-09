import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv("kumasi_transport_ml_ready.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

# These are the variables the AI will use to predict demand
features = [
    "hour",
    "location",
    "transport_mode",
    "weather",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "seasonal_factor",
    "traffic_index",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_rolling_3h",
    "demand_rolling_24h"
]

# Target variable
target = "boarding_count"


X = df[features]
y = df[target]


print("\nFeatures being used:")
for feature in features:
    print("-", feature)

print("\nTarget variable:")
print("-", target)


# ============================================================
# 3. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "location",
    "transport_mode",
    "weather"
]

numerical_features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "seasonal_factor",
    "traffic_index",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_rolling_3h",
    "demand_rolling_24h"
]


# ============================================================
# 4. ENCODE CATEGORICAL DATA
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 5. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. CREATE MACHINE LEARNING PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

print("\nTraining Random Forest model...")
print("Please wait...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = pipeline.predict(X_test)


# ============================================================
# 10. CALCULATE MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("========================================")


# ============================================================
# 11. SHOW SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Demand": y_test.values,
    "Predicted Demand": y_pred
})

print("\nSample Predictions:")
print(results.head(20))


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

results.to_csv(
    "transport_demand_predictions.csv",
    index=False
)

print("\nPredictions saved as:")
print("transport_demand_predictions.csv")


# ============================================================
# 13. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3
)

plt.xlabel("Actual Passenger Demand")
plt.ylabel("Predicted Passenger Demand")
plt.title("Actual vs Predicted Public Transport Demand")

# Perfect prediction reference line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()


print("\nGraph saved as:")
print("actual_vs_predicted.png")

print("\nMODEL TRAINING COMPLETE!")