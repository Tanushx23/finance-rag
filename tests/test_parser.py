"""
Tests for core/parser.py -- specifically the dropna() bug we found and
fixed: the original code did df.dropna() with no subset, which deleted an
entire transaction if ANY column was blank, including description.
"""
import io
import pandas as pd
import pytest

from core.parser import load_transactions_df, build_chunks, REQUIRED_COLUMNS


def _csv(text: str):
    return io.BytesIO(text.strip().encode("utf-8"))


def test_parses_all_valid_rows():
    csv = _csv("""
date,amount,category,description
2024-01-05,500,Food,Zomato order
2024-01-06,1200,Shopping,Amazon purchase
""")
    df = load_transactions_df(csv)
    assert len(df) == 2
    assert df["amount"].sum() == 1700


def test_blank_description_does_not_drop_the_row():
    """Regression test for the original bug: a blank description used to
    delete the whole transaction via an unscoped dropna()."""
    csv = _csv("""
date,amount,category,description
2024-01-05,500,Food,
2024-01-06,1200,Shopping,Amazon purchase
""")
    df = load_transactions_df(csv)
    assert len(df) == 2  # both rows kept, not just the one with a description


def test_missing_amount_drops_only_that_row():
    csv = _csv("""
date,amount,category,description
2024-01-05,,Food,Zomato order
2024-01-06,1200,Shopping,Amazon purchase
""")
    df = load_transactions_df(csv)
    assert len(df) == 1
    assert df.iloc[0]["amount"] == 1200


def test_missing_required_column_raises_clear_error():
    csv = _csv("""
date,amount,description
2024-01-05,500,Zomato order
""")
    with pytest.raises(ValueError, match="category"):
        load_transactions_df(csv)


def test_build_chunks_formats_expected_text():
    csv = _csv("""
date,amount,category,description
2024-01-05,500,Food,Zomato order
""")
    df = load_transactions_df(csv)
    chunks = build_chunks(df)
    assert chunks == ["On 2024-01-05, spent 500 on Food — Zomato order"]