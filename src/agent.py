"""
agent.py
--------
Autonomous Inventory Management Agent.

Wraps the existing forecasting pipeline with an agentic loop powered by
the Claude API's tool_use feature. The agent:
  1. Perceives  — receives the final forecast DataFrame
  2. Reasons    — decides which items need action
  3. Acts       — calls tools (generate PO, flag for review, etc.)
  4. Reflects   — checks completeness before stopping
  5. Reports    — returns a plain-English narrative + action log

No changes required to preprocessing.py, forecasting.py, or inventory.py.
"""

import json
import os
from typing import Generator

import anthropic

from src.agent_tools import (
    TOOL_DEFINITIONS,
    analyse_inventory_status,
    generate_purchase_order,
    flag_for_review,
    get_item_sales_trend,
    generate_manager_alert,
    clear_action_log,
    get_action_log,
)

# ---------------------------------------------------------------------------
# System prompt — defines the agent's persona, goals, and constraints
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an Autonomous Inventory Management Agent for a retail business.

Your goal is to analyse forecast data and take smart, conservative inventory actions.

## Your tools
- analyse_inventory_status   → always call this FIRST to get a structured overview
- get_item_sales_trend       → dig into a specific item before acting on it
- generate_purchase_order    → place a reorder for confirmed understocks
- flag_for_review            → escalate ambiguous or anomalous items to a human
- generate_manager_alert     → notify store managers of important changes

## Decision rules
1. Always call analyse_inventory_status first — never act without data.
2. For each understock item:
   - If understock_units > 10: generate a purchase order immediately.
   - If understock_units is 1-10: call get_item_sales_trend first. 
     If trend is "rising" → order. If "volatile" or "anomalous" → flag_for_review.
3. For volatile items (rolling_std_7 unusually high): always flag_for_review.
4. After taking actions for a store, call generate_manager_alert for that store.
5. After handling all items, reflect: did you address everything? If not, continue.
6. Do not hallucinate data. Only reason from what the tools return.
7. Be conservative — when in doubt, flag rather than order.

## Response style
- Be concise and business-focused.
- Always end with a clear summary: what you found, what you did, what needs human attention.
- Use plain English — assume the reader is a store manager, not a data scientist.
- Mention specific item IDs, quantities, and PO numbers.
"""

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def _dispatch_tool(tool_name: str, tool_input: dict,
                   forecast_json: str) -> str:
    """Route a tool call from the agent to the correct Python function."""

    # Inject forecast_json automatically where needed
    if tool_name in ("analyse_inventory_status", "get_item_sales_trend"):
        tool_input.setdefault("forecast_json", forecast_json)

    if tool_name == "analyse_inventory_status":
        return analyse_inventory_status(tool_input["forecast_json"])

    if tool_name == "generate_purchase_order":
        return generate_purchase_order(
            store_id=tool_input["store_id"],
            item_id=tool_input["item_id"],
            quantity=tool_input["quantity"],
            reason=tool_input["reason"],
        )

    if tool_name == "flag_for_review":
        return flag_for_review(
            store_id=tool_input["store_id"],
            item_id=tool_input["item_id"],
            reason=tool_input["reason"],
            severity=tool_input.get("severity", "medium"),
        )

    if tool_name == "get_item_sales_trend":
        return get_item_sales_trend(
            forecast_json=tool_input["forecast_json"],
            store_id=tool_input["store_id"],
            item_id=tool_input["item_id"],
        )

    if tool_name == "generate_manager_alert":
        return generate_manager_alert(
            store_id=tool_input["store_id"],
            summary=tool_input["summary"],
            urgency=tool_input.get("urgency", "normal"),
        )

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ---------------------------------------------------------------------------
# Agentic loop — autonomous run
# ---------------------------------------------------------------------------
def run_agent(
    forecast_df,
    api_key: str,
    max_iterations: int = 10,
) -> dict:
    """
    Run the autonomous agent on a forecast DataFrame.

    Returns:
        {
          "final_response": str,   # plain-English narrative
          "action_log": list,      # all tool calls made
          "iterations": int,       # how many reasoning loops ran
        }
    """
    clear_action_log()
    client = anthropic.Anthropic(api_key=api_key)

    # Serialize the forecast for tool injection
    forecast_json = forecast_df.to_json(orient="records", date_format="iso")

    messages = [
        {
            "role": "user",
            "content": (
                "I've just run the demand forecasting pipeline. "
                "Please analyse the results and take appropriate inventory actions. "
                f"The forecast covers {len(forecast_df)} rows across "
                f"{forecast_df['store_id'].nunique()} store(s) and "
                f"{forecast_df['item_id'].nunique()} item(s)."
            ),
        }
    ]

    final_response = ""
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Collect any text blocks
        for block in response.content:
            if hasattr(block, "text"):
                final_response = block.text

        # Stop if no more tool calls
        if response.stop_reason == "end_turn":
            break

        # Process tool calls
        if response.stop_reason == "tool_use":
            # Append assistant message
            messages.append({"role": "assistant", "content": response.content})

            # Build tool results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _dispatch_tool(
                        block.name, block.input, forecast_json
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    return {
        "final_response": final_response,
        "action_log": get_action_log(),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Chat mode — conversational follow-up with memory
# ---------------------------------------------------------------------------
def chat_with_agent(
    user_message: str,
    conversation_history: list,
    forecast_df,
    api_key: str,
) -> tuple[str, list]:
    """
    Single-turn conversational exchange. Maintains full history.

    Returns:
        (assistant_reply: str, updated_history: list)
    """
    client = anthropic.Anthropic(api_key=api_key)
    forecast_json = forecast_df.to_json(orient="records", date_format="iso")

    # Append user message
    history = conversation_history + [
        {"role": "user", "content": user_message}
    ]

    max_turns = 6
    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=history,
        )

        if response.stop_reason == "end_turn":
            reply = " ".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            history.append({"role": "assistant", "content": reply})
            return reply, history

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _dispatch_tool(
                        block.name, block.input, forecast_json
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            history.append({"role": "user", "content": tool_results})

    # Fallback
    return "Agent reached max turns without a final response.", history
