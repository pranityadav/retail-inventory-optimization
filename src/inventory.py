import numpy as np

def inventory_decision(
    df,
    safety_stock,
    understock_cost,
    overstock_cost
):
    df = df.copy()

    df["recommended_inventory"] = (
        df["predicted_sales"] + safety_stock
    ).round()

    df["understock_units"] = np.maximum(
        df["sales"] - df["recommended_inventory"], 0
    )

    df["overstock_units"] = np.maximum(
        df["recommended_inventory"] - df["sales"], 0
    )

    df["estimated_cost"] = (
        df["understock_units"] * understock_cost +
        df["overstock_units"] * overstock_cost
    )

    return df