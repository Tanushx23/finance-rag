import pandas as pd

REQUIRED_COLUMNS = ["date", "amount", "category"]


def load_transactions_df(file) -> pd.DataFrame:
    """Parse the CSV into a clean, validated DataFrame.

    This is the structured form used for exact computation (sums, filters
    by category/date range).
    """
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["description"] = df.get("description", "").fillna("")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["category"] = df["category"].astype(str).str.strip()

    return df.reset_index(drop=True)


def build_chunks(df: pd.DataFrame) -> list[str]:
    """Build the text chunks used for embedding + semantic search, from an
    already-parsed DataFrame (so we never read the uploaded file twice)."""
    chunks = []
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        chunk = (
            f"On {date_str}, spent {row['amount']} on {row['category']}"
            + (f" — {row['description']}" if row["description"] else "")
        )
        chunks.append(chunk)
    return chunks


def load_transactions(file) -> list[str]:
    """Back-compat helper: parse + build chunks in one call."""
    df = load_transactions_df(file)
    return build_chunks(df)