import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv("kumasi_transport_ml_ready.csv")

print("Dataset loaded:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

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

target = "boarding_count"


X = df[features]
y = df[target]


# ============================================================
# 3. CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "location",
    "transport_mode",
    "weather"
]


# ============================================================
# 4. NUMERICAL FEATURES
# ============================================================

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
# 5. PREPROCESSOR
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
# 6. RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining final model...")

pipeline.fit(X, y)

print("Training completed!")


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "transport_demand_model.pkl"
)

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print("File:")
print("transport_demand_model.pkl")