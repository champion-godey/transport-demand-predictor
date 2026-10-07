import pandas as pd
import matplotlib.pyplot as plt

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

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
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
# 4. PREPROCESSING
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
# 5. CREATE MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 7. PREPROCESS DATA
# ============================================================

print("\nPreparing data...")

X_train_processed = preprocessor.fit_transform(X_train)


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("Training Random Forest...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed!")


# ============================================================
# 9. GET FEATURE NAMES
# ============================================================

encoded_features = preprocessor.named_transformers_[
    "categorical"
].get_feature_names_out(categorical_features)

feature_names = list(encoded_features) + numerical_features


# ============================================================
# 10. GET FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


# ============================================================
# 11. SORT FEATURES
# ============================================================

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(
    importance_df.to_string(index=False)
)


# ============================================================
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("feature_importance.csv")


# ============================================================
# 13. DISPLAY TOP 15 FEATURES
# ============================================================

top_features = importance_df.head(15)

print("\n========================================")
print("TOP 15 FEATURES")
print("========================================")

print(
    top_features.to_string(index=False)
)


# ============================================================
# 14. CREATE BAR CHART
# ============================================================

plt.figure(figsize=(12, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Features Influencing Public Transport Demand"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()


print("\nChart saved as:")
print("feature_importance.png")

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETE!")