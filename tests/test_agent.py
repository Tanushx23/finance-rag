"""
Regression tests for core/agent.py's deterministic computation logic.
Each of these maps to a real bug found while manually testing the app --
see comments for what would have happened before the fix.
"""
import io
import pytest

from core.parser import load_transactions_df
from core.agent import _compute_total_spent, _resolve_category, _get_top_transaction, _get_category_breakdown


SAMPLE_CSV = """date,amount,category,description
2024-01-05,500,Food,Zomato order
2024-01-06,1200,Shopping,Amazon purchase
2024-01-08,350,Food,Swiggy delivery
2024-01-10,2000,Shopping,Myntra clothes
2024-01-12,600,Food,Restaurant dinner
2024-01-15,800,Transport,Uber rides
2024-01-18,1500,Entertainment,Netflix + BookMyShow
2024-01-20,400,Food,Grocery store
2024-01-22,300,Transport,Metro card recharge
2024-01-25,1000,Shopping,Flipkart order
"""


@pytest.fixture
def df():
    return load_transactions_df(io.BytesIO(SAMPLE_CSV.encode("utf-8")))

def test_parser_handles_missing_description_column():
    """Regression test: df.get('description', '').fillna('') crashed with
    AttributeError when the description column was entirely absent, since
    df.get() returns a plain string fallback (not a Series) in that case,
    and strings have no .fillna(). CSVs with only date/amount/category
    (no description) must parse successfully with empty descriptions."""
    csv_without_description = "date,amount,category\n2024-01-05,500,Food\n"
    df = load_transactions_df(io.BytesIO(csv_without_description.encode("utf-8")))
    assert df["description"].tolist() == [""]
    
def test_category_total_is_correct(df):
    """The original bug: the LLM was asked to add these numbers itself and
    got 1750, then self-corrected mid-sentence to 1850. This must now be
    computed in pandas, not guessed by a model."""
    result = _compute_total_spent(df, category="Food")
    assert result["total"] == 1850.0
    assert result["transaction_count"] == 4
    assert result["total_formatted"] == "₹1,850"


def test_total_with_no_filters(df):
    result = _compute_total_spent(df)
    assert result["total"] == 8650.0
    assert result["transaction_count"] == 10


def test_date_range_filter_is_correct(df):
    """Regression test: the model once defaulted to the CURRENT year
    (2026) instead of the data's actual year (2024) when the question
    omitted a year, silently returning $0. This test locks in the correct
    numeric answer for the known date range."""
    result = _compute_total_spent(df, start_date="2024-01-01", end_date="2024-01-15")
    assert result["total"] == 5450.0
    assert result["transaction_count"] == 6


def test_category_typo_resolves_via_case_insensitive_fuzzy_match(df):
    """Casing/typo variants should resolve to the real category. Note: true
    synonyms like 'groceries' -> 'Food' are the LLM's job (it sees the real
    category list in the tool schema) -- this fallback only catches string-
    level typos/casing, not semantic mapping."""
    result = _compute_total_spent(df, category="fod")  # lowercase + typo
    assert result["category"] == "Food"
    assert result["total"] == 1850.0


def test_unrecognized_category_fails_clearly_instead_of_returning_zero(df):
    """A category with no reasonable match should surface an explicit
    error (with the real category list) rather than a confident-looking
    but meaningless $0 total."""
    result = _compute_total_spent(df, category="Bitcoin")
    assert "error" in result
    assert "available_categories" in result


def test_resolve_category_exact_match():
    assert _resolve_category("Food", ["Food", "Shopping"]) == "Food"


def test_resolve_category_case_insensitive_typo():
    assert _resolve_category("food", ["Food", "Shopping"]) == "Food"
    assert _resolve_category("fod", ["Food", "Shopping"]) == "Food"


def test_resolve_category_no_match_returns_none():
    assert _resolve_category("Bitcoin", ["Food", "Shopping"]) is None

def test_average_is_computed_exactly_not_by_the_model(df):
    """The model once divided total/count itself in its final answer text
    (which happened to come out clean: 8650/10 = 865) instead of using a
    computed field. This locks in a case with messier division (1850/4 =
    462.5) so the average is always exact, not dependent on the model's
    own arithmetic."""
    result = _compute_total_spent(df, category="Food")
    assert result["average"] == 462.5
    assert result["average_formatted"] == "₹462"


def test_average_with_no_filters(df):
    result = _compute_total_spent(df)
    assert result["average"] == 865.0


def test_most_expensive_transaction_overall(df):
    """Before this tool existed, 'most expensive purchase' had to go through
    semantic_search, which ranks by meaning, not amount -- it happened to
    get this right by luck earlier in manual testing, but wasn't
    guaranteed to. This tool makes it deterministic."""
    result = _get_top_transaction(df, direction="max")
    assert result["amount"] == 2000.0
    assert result["description"] == "Myntra clothes"


def test_cheapest_transaction_overall(df):
    result = _get_top_transaction(df, direction="min")
    assert result["amount"] == 300.0


def test_most_expensive_within_category(df):
    result = _get_top_transaction(df, direction="max", category="Food")
    assert result["amount"] == 600.0
    assert result["category"] == "Food"


def test_top_transaction_with_no_matches_returns_error(df):
    result = _get_top_transaction(df, direction="max", start_date="2030-01-01")
    assert "error" in result

def test_category_breakdown_ranks_correctly(df):
    """Before this tool existed, 'which category do I spend most on' had
    no reliable path -- the agent would either need several separate
    compute_total_spent calls or guess via semantic_search. This locks in
    the correct ranked breakdown against known totals."""
    result = _get_category_breakdown(df)
    assert result["top_category"] == "Shopping"
    assert result["breakdown"][0]["total"] == 4200.0
    assert result["breakdown"][-1]["category"] == "Transport"
    total_pct = sum(item["percent_of_total"] for item in result["breakdown"])
    assert abs(total_pct - 100.0) < 0.5


def test_category_breakdown_with_no_matches_returns_error(df):
    result = _get_category_breakdown(df, start_date="2030-01-01")
    assert "error" in result