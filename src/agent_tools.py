"""
agent_tools.py
--------------
Tool functions the inventory agent can call autonomously.
Each function is also registered as a Claude tool definition (see TOOL_DEFINITIONS).
"""

import json
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# In-memory log of all agent actions this session
# ---------------------------------------------------------------------------
_action_log: list[dict] = []


def _log(tool: str, inputs: dict, result: dict):
    _action_log.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "tool": tool,
        "inputs": inputs,
        "result": result,
    })


def get_action_log() -> list[dict]:
    return _action_log.copy()


def clear_action_log():
    _action_log.clear()


# ---------------------------------------------------------------------------
# Tool 1 – analyse_inventory_status
# ---------------------------------------------------------------------------
def analyse_inventory_status(forecast_json: str) -> str:
    """
    Parse the forecast DataFrame (passed as JSON) and return a structured
    summary: critical understocks, overstocks, and volatility flags.
    """
    try:
        df = pd.read_json(forecast_json, orient="records")
        required = {"store_id", "item_id", "predicted_sales",
                    "recommended_inventory", "understock_units",
                    "overstock_units", "estimated_cost"}
        missing = required - set(df.columns)
        if missing:
            return json.dumps({"error": f"Missing columns: {missing}"})

        critical = df[df["understock_units"] > 0][
            ["store_id", "item_id", "understock_units",
             "recommended_inventory", "estimated_cost"]
        ].sort_values("understock_units", ascending=False).head(10)

        overstock = df[df["overstock_units"] > 5][
            ["store_id", "item_id", "overstock_units", "estimated_cost"]
        ].sort_values("overstock_units", ascending=False).head(10)

        volatility = []
        if "rolling_std_7" in df.columns:
            mean_std = df["rolling_std_7"].mean()
            volatile = df[df["rolling_std_7"] > mean_std * 1.5][
                ["store_id", "item_id", "rolling_std_7"]
            ].drop_duplicates().head(5)
            volatility = volatile.to_dict(orient="records")

        result = {
            "total_rows": len(df),
            "total_estimated_cost": round(float(df["estimated_cost"].sum()), 2),
            "critical_understocks": critical.to_dict(orient="records"),
            "overstock_items": overstock.to_dict(orient="records"),
            "volatile_items": volatility,
        }
        _log("analyse_inventory_status", {}, result)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 2 – generate_purchase_order
# ---------------------------------------------------------------------------
def generate_purchase_order(store_id: str, item_id: str,
                             quantity: int, reason: str) -> str:
    """
    Draft a purchase order for a given store/item/quantity.
    Returns a PO reference number and summary.
    """
    po_number = f"PO-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{store_id}-{item_id}"
    result = {
        "po_number": po_number,
        "store_id": store_id,
        "item_id": item_id,
        "quantity_ordered": quantity,
        "reason": reason,
        "status": "DRAFT",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _log("generate_purchase_order",
         {"store_id": store_id, "item_id": item_id,
          "quantity": quantity, "reason": reason},
         result)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 3 – flag_for_review
# ---------------------------------------------------------------------------
def flag_for_review(store_id: str, item_id: str, reason: str,
                    severity: str = "medium") -> str:
    """
    Flag a store/item combination for human review instead of taking
    autonomous action. severity: low | medium | high
    """
    result = {
        "flagged": True,
        "store_id": store_id,
        "item_id": item_id,
        "reason": reason,
        "severity": severity,
        "flagged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": (
            f"{item_id} at {store_id} has been flagged ({severity} severity). "
            f"Reason: {reason}. Please review before any reorder is placed."
        ),
    }
    _log("flag_for_review",
         {"store_id": store_id, "item_id": item_id,
          "reason": reason, "severity": severity},
         result)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 4 – get_item_sales_trend
# ---------------------------------------------------------------------------
def get_item_sales_trend(forecast_json: str,
                          store_id: str, item_id: str) -> str:
    """
    Return the recent sales trend for a specific store/item combination,
    including lag values and rolling statistics.
    """
    try:
        df = pd.read_json(forecast_json, orient="records")
        mask = (df["store_id"].astype(str) == str(store_id)) & \
               (df["item_id"].astype(str) == str(item_id))
        subset = df[mask]
        if subset.empty:
            return json.dumps({"error": f"No data for {store_id}/{item_id}"})

        row = subset.iloc[-1]
        trend_cols = ["lag_1", "lag_7", "lag_14",
                      "rolling_mean_7", "rolling_std_7",
                      "predicted_sales", "recommended_inventory"]
        trend = {c: round(float(row[c]), 2)
                 for c in trend_cols if c in row.index}
        trend.update({"store_id": store_id, "item_id": item_id})

        # Simple trend direction
        if "lag_1" in trend and "rolling_mean_7" in trend:
            diff = trend["lag_1"] - trend["rolling_mean_7"]
            trend["trend_direction"] = (
                "rising" if diff > 1 else "falling" if diff < -1 else "stable"
            )

        _log("get_item_sales_trend",
             {"store_id": store_id, "item_id": item_id}, trend)
        return json.dumps(trend)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 5 – generate_manager_alert
# ---------------------------------------------------------------------------
def generate_manager_alert(store_id: str, summary: str,
                             urgency: str = "normal") -> str:
    """
    Generate a formatted manager alert message for a store.
    urgency: low | normal | urgent
    """
    emoji = {"low": "ℹ️", "normal": "⚠️", "urgent": "🚨"}.get(urgency, "⚠️")
    message = (
        f"{emoji} Inventory Alert — {store_id}\n"
        f"Date: {datetime.now().strftime('%d %b %Y, %H:%M')}\n\n"
        f"{summary}\n\n"
        f"This alert was generated automatically by the Inventory Management Agent."
    )
    result = {
        "store_id": store_id,
        "urgency": urgency,
        "message": message,
        "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _log("generate_manager_alert",
         {"store_id": store_id, "urgency": urgency}, result)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Claude Tool Definitions (passed to the API)
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "analyse_inventory_status",
        "description": (
            "Analyse the full forecast DataFrame to identify critical understocks, "
            "overstocks, and volatile items. Always call this first before taking "
            "any other action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast_json": {
                    "type": "string",
                    "description": "The forecast DataFrame serialized as JSON (orient='records')."
                }
            },
            "required": ["forecast_json"]
        }
    },
    {
        "name": "generate_purchase_order",
        "description": (
            "Draft a purchase order for a specific store and item. "
            "Use when understock is confirmed and the cause is not ambiguous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store identifier."},
                "item_id":  {"type": "string", "description": "Item/product identifier."},
                "quantity": {"type": "integer", "description": "Units to order."},
                "reason":   {"type": "string", "description": "One-line reason for this order."}
            },
            "required": ["store_id", "item_id", "quantity", "reason"]
        }
    },
    {
        "name": "flag_for_review",
        "description": (
            "Flag a store/item for human review instead of acting autonomously. "
            "Use when data looks anomalous, when volatility is unusually high, "
            "or when you are uncertain about the right action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "item_id":  {"type": "string"},
                "reason":   {"type": "string", "description": "Why this item needs human review."},
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How urgently a human should review this."
                }
            },
            "required": ["store_id", "item_id", "reason", "severity"]
        }
    },
    {
        "name": "get_item_sales_trend",
        "description": (
            "Get detailed lag features, rolling statistics, and trend direction "
            "for a specific store/item combination. Use to investigate anomalies "
            "before deciding whether to order or flag."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast_json": {"type": "string", "description": "Forecast DataFrame as JSON."},
                "store_id": {"type": "string"},
                "item_id":  {"type": "string"}
            },
            "required": ["forecast_json", "store_id", "item_id"]
        }
    },
    {
        "name": "generate_manager_alert",
        "description": (
            "Generate a formatted manager alert message for a store. "
            "Use after taking actions so the store manager knows what happened."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string"},
                "summary":  {"type": "string", "description": "2-3 sentence plain-English summary of what the agent found and did."},
                "urgency":  {"type": "string", "enum": ["low", "normal", "urgent"]}
            },
            "required": ["store_id", "summary", "urgency"]
        }
    }
]
