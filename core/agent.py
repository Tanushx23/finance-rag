"""
Small tool-calling agent layer.

Problem this solves: asking the LLM to add up numbers from retrieved text
chunks is unreliable (it can miscount, as our own test just showed:
"1750... 400+600+350+500=1850... the correct total is 1850").

Fix: give the model tools. For anything involving a sum/total/count, it
calls `compute_total_spent`, which does the arithmetic in pandas (always
correct). For "most expensive"/"cheapest" questions, it calls
`get_top_transaction`. For genuinely open-ended/fuzzy questions, it calls
`semantic_search`, which reuses the existing FAISS vector store. The model
never does the math itself -- it only decides which tool to call and then
phrases the (already-computed) result in natural language.
"""

import difflib
import json
import logging
import os

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from core.embeddings import get_embeddings

load_dotenv()

logger = logging.getLogger("finance_rag")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# llama-3.3-70b-versatile is being deprecated by Groq (announced June 2026)
# and has a known bug where it sometimes emits tool calls as malformed
# <function=...></function> text instead of a proper structured call,
# which Groq's API then rejects with a 400 tool_use_failed error. Groq's
# own recommended replacement is openai/gpt-oss-120b.
MODEL = "openai/gpt-oss-120b"


def _build_tools(categories: list[str], min_date: str, max_date: str) -> list[dict]:
    """Build the tool schema with the ACTUAL category values and date range
    present in this user's data. Two real bugs this fixes:

    1. Category was previously a free-text hint, so the model had to guess
       the exact string -- typos/synonyms silently matched zero rows.
       Constraining to a real enum fixes that (see _resolve_category too).

    2. Without knowing what date range the data actually covers, a question
       like "between Jan 1 and Jan 15" (no year given) led the model to
       assume the CURRENT year rather than the year the data is actually
       in -- silently returning $0 instead of matching anything. Telling
       it the real min/max dates fixes this."""
    return [
        {
            "type": "function",
            "function": {
                "name": "compute_total_spent",
                "description": (
                    "Compute the exact total amount spent, transaction count, and "
                    "average amount per transaction, optionally filtered by "
                    "category and/or date range. ALWAYS use this for any question "
                    "asking for a sum, total, count, or average -- never add, "
                    "count, or divide numbers yourself, and never use "
                    "semantic_search for counting (it only returns a limited "
                    "sample of matches, not the full count)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": ["string", "null"],
                            "enum": categories + [None],
                            "description": (
                                "Filter to this exact category. Map the user's wording "
                                "onto the closest matching category from this list "
                                "(e.g. 'groceries' or 'eating out' -> 'Food'). Use null "
                                "for all categories."
                            ),
                        },
                        "start_date": {
                            "type": ["string", "null"],
                            "description": (
                                f"YYYY-MM-DD, inclusive lower bound. The transaction data "
                                f"spans from {min_date} to {max_date} -- if the user gives "
                                f"a date/month without a year, use the year from this range, "
                                f"NOT the current calendar year. Use null for no lower bound."
                            ),
                        },
                        "end_date": {
                            "type": ["string", "null"],
                            "description": (
                                f"YYYY-MM-DD, inclusive upper bound. The transaction data "
                                f"spans from {min_date} to {max_date} -- if the user gives "
                                f"a date/month without a year, use the year from this range, "
                                f"NOT the current calendar year. Use null for no upper bound."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_transaction",
                "description": (
                    "Find the single largest or smallest transaction by amount, "
                    "optionally filtered by category and/or date range. Use this "
                    "for 'most expensive', 'biggest purchase', 'cheapest', "
                    "'smallest transaction' style questions -- do NOT use "
                    "semantic_search for this, since it ranks by meaning, not "
                    "by amount, and can return the wrong transaction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["max", "min"],
                            "description": "'max' for most expensive/largest, 'min' for cheapest/smallest.",
                        },
                        "category": {
                            "type": ["string", "null"],
                            "enum": categories + [None],
                            "description": "Filter to this exact category. Use null for all categories.",
                        },
                        "start_date": {
                            "type": ["string", "null"],
                            "description": f"YYYY-MM-DD inclusive lower bound. Data spans {min_date} to {max_date}. Use null for no lower bound.",
                        },
                        "end_date": {
                            "type": ["string", "null"],
                            "description": f"YYYY-MM-DD inclusive upper bound. Data spans {min_date} to {max_date}. Use null for no upper bound.",
                        },
                    },
                    "required": ["direction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_category_breakdown",
                "description": (
                    "Get total spending per category, ranked highest to lowest, "
                    "optionally filtered by date range. Use this for 'which "
                    "category do I spend the most on', 'breakdown by category', "
                    "or 'top spending categories' style questions -- do NOT "
                    "answer these by calling compute_total_spent multiple times "
                    "or by guessing from semantic_search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": ["string", "null"],
                            "description": f"YYYY-MM-DD inclusive lower bound. Data spans {min_date} to {max_date}. Use null for no lower bound.",
                        },
                        "end_date": {
                            "type": ["string", "null"],
                            "description": f"YYYY-MM-DD inclusive upper bound. Data spans {min_date} to {max_date}. Use null for no upper bound.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "Search transactions by meaning for open-ended or fuzzy "
                    "questions that are NOT a total or count, e.g. 'what's my "
                    "most unusual purchase' or 'did I spend anything on gifts'. "
                    "Do NOT use this to count or sum transactions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the search query"}
                    },
                    "required": ["query"],
                },
            },
        },
    ]


SYSTEM_PROMPT = """You are a personal finance assistant for Indian Rupee (INR)
transactions. Always use the RUPEE symbol (₹), never $, and use the exact
formatted strings from tool results (e.g. total_formatted, average_formatted)
rather than reformatting or recalculating numbers yourself.
You have four tools: `compute_total_spent` for any sum/total/count/average/
"how much"/"how many" question, `get_top_transaction` for "most expensive"/
"biggest"/"cheapest"/"smallest" questions, `get_category_breakdown` for
"which category do I spend most on" / spending-breakdown questions, and
`semantic_search` for open-ended questions about specific transactions
(never for counting, ranking, or breakdowns -- it only returns a limited
sample ranked by meaning).
Always call a tool rather than computing totals, counts, averages, rankings,
breakdowns, or dates yourself. After you get a tool result, answer the
user's question clearly and concisely based only on that result."""


def _resolve_category(category: str, valid_categories: list[str]) -> str | None:
    """Safety net behind the schema enum: if the model still sends something
    that isn't an exact match (e.g. a Groq model that ignores enum, a typo,
    or different casing), fuzzy-match it to the closest real category
    instead of silently matching zero rows. Returns None if nothing is
    close enough.

    Note: this is a STRING-similarity fallback (difflib), not a semantic
    one -- it catches "fod"/"FOOD" typos/casing, but true synonyms like
    "groceries" -> "Food" rely on the LLM itself picking the right enum
    value (it's told the real category list in the tool schema), not on
    this function. This is just the safety net for when that doesn't
    happen cleanly."""
    if category in valid_categories:
        return category

    lowered_map = {c.lower(): c for c in valid_categories}
    if category.lower() in lowered_map:
        return lowered_map[category.lower()]

    matches = difflib.get_close_matches(
        category.lower(), list(lowered_map.keys()), n=1, cutoff=0.6
    )
    return lowered_map[matches[0]] if matches else None


def _compute_total_spent(df: pd.DataFrame, category=None, start_date=None, end_date=None) -> dict:
    filtered = df
    resolved_category = None

    if category:
        valid_categories = df["category"].unique().tolist()
        resolved_category = _resolve_category(category, valid_categories)
        if resolved_category is None:
            return {
                "error": f"'{category}' doesn't match any known category.",
                "available_categories": valid_categories,
            }
        filtered = filtered[filtered["category"] == resolved_category]

    if start_date:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]

    total = float(filtered["amount"].sum())
    count = int(len(filtered))
    # Compute the average here in pandas rather than letting the model
    # divide total/count itself in its final answer -- simple division
    # happened to come out clean in testing ($8650/10 = $865), but nothing
    # guaranteed the model would do that arithmetic correctly every time
    # (same root problem as the original "add up these chunks" bug).
    average = total / count if count > 0 else 0.0

    return {
        "total": total,
        "total_formatted": f"₹{total:,.0f}",
        "average": average,
        "average_formatted": f"₹{average:,.0f}",
        "transaction_count": count,
        "category": resolved_category or "all categories",
        "start_date": start_date,
        "end_date": end_date,
    }


def _get_top_transaction(df: pd.DataFrame, direction="max", category=None, start_date=None, end_date=None) -> dict:
    filtered = df
    resolved_category = None

    if category:
        valid_categories = df["category"].unique().tolist()
        resolved_category = _resolve_category(category, valid_categories)
        if resolved_category is None:
            return {
                "error": f"'{category}' doesn't match any known category.",
                "available_categories": valid_categories,
            }
        filtered = filtered[filtered["category"] == resolved_category]

    if start_date:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]

    if len(filtered) == 0:
        return {"error": "No transactions match those filters."}

    idx = filtered["amount"].idxmax() if direction == "max" else filtered["amount"].idxmin()
    row = filtered.loc[idx]

    return {
        "amount": float(row["amount"]),
        "amount_formatted": f"₹{row['amount']:,.0f}",
        "date": row["date"].strftime("%Y-%m-%d"),
        "category": row["category"],
        "description": row["description"],
    }

def _get_category_breakdown(df: pd.DataFrame, start_date=None, end_date=None) -> dict:
    filtered = df
    if start_date:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]

    if len(filtered) == 0:
        return {"error": "No transactions match those filters."}

    grand_total = float(filtered["amount"].sum())
    grouped = (
        filtered.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    breakdown = [
        {
            "category": category,
            "total": float(total),
            "total_formatted": f"₹{total:,.0f}",
            "percent_of_total": round(float(total) / grand_total * 100, 1) if grand_total > 0 else 0.0,
        }
        for category, total in grouped.items()
    ]

    return {
        "breakdown": breakdown,
        "top_category": breakdown[0]["category"] if breakdown else None,
        "grand_total_formatted": f"₹{grand_total:,.0f}",
    }


def _semantic_search(vector_store, query: str, k: int = 8) -> dict:
    # k bumped from 5 -> 8: with only 5, a genuinely broad question ("show
    # me all my food purchases") could silently truncate real matches out
    # of the result. 8 is still cheap for FAISS and gives the LLM more to
    # work with for open-ended questions.
    query_embedding = get_embeddings([query])[0]
    results = vector_store.search(query_embedding, k=k)
    return {"matching_transactions": results}


def answer_question(question: str, df: pd.DataFrame, vector_store) -> dict:
    """Run the agent loop: model picks a tool, we execute it locally, model
    phrases the final answer. Returns dict with `answer` and `tool_used` so
    the API/UI can show which path was taken (nice for debugging + demos)."""

    categories = df["category"].unique().tolist()
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    tools = _build_tools(categories, min_date, max_date)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Occasionally a model emits a malformed tool call that Groq's API
    # rejects with a 400 tool_use_failed error (not our bug -- a model
    # generation quirk). Retry once before giving up, since it's usually
    # a one-off rather than a persistent failure.
    first_response = None
    last_error = None
    for attempt in range(2):
        try:
            first_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Tool-calling attempt {attempt + 1} failed: {e}")

    if first_response is None:
        raise last_error

    choice = first_response.choices[0].message
    tool_calls = choice.tool_calls

    if not tool_calls:
        return {"answer": choice.content, "tool_used": None}

    messages.append(choice)
    tool_used = None

    for call in tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments or "{}")
        tool_used = name

        if name == "compute_total_spent":
            result = _compute_total_spent(df, **args)
        elif name == "get_top_transaction":
            result = _get_top_transaction(df, **args)
        elif name == "get_category_breakdown":
            result = _get_category_breakdown(df, **args)
        elif name == "semantic_search":
            result = _semantic_search(vector_store, **args)
        else:
            result = {"error": f"Unknown tool {name}"}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return {
        "answer": final_response.choices[0].message.content,
        "tool_used": tool_used,
    }