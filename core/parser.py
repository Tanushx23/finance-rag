import pandas as pd

def load_transactions(file) -> list[str]:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna()

    chunks = []
    for _, row in df.iterrows():
        chunk = f"On {row.get('date', 'unknown date')}, spent {row.get('amount', '?')} on {row.get('category', 'unknown category')} — {row.get('description', '')}"
        chunks.append(chunk)

    return chunks