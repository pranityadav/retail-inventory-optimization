import joblib

FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_std_7",
    "dayofweek", "week", "month"
]

def load_model(path: str):
    return joblib.load(path)

def predict(model, df):
    X = df[FEATURES]
    df["predicted_sales"] = model.predict(X)
    return df