import pandas as pd

def validate_input(df: pd.DataFrame):
    required_cols = {"date", "store_id", "item_id", "sales"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"])

    df["dayofweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    df["lag_1"] = df.groupby(["store_id", "item_id"])["sales"].shift(1)
    df["lag_7"] = df.groupby(["store_id", "item_id"])["sales"].shift(7)
    df["lag_14"] = df.groupby(["store_id", "item_id"])["sales"].shift(14)

    df["rolling_mean_7"] = (
        df.groupby(["store_id", "item_id"])["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["rolling_std_7"] = (
        df.groupby(["store_id", "item_id"])["sales"]
        .shift(1)
        .rolling(7)
        .std()
    )

    df = df.dropna()
    return df