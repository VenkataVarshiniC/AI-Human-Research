"""
data_cleaning.py
----------------
Loads, validates, and cleans the AI productivity study dataset.
Outputs a clean DataFrame and derived feature columns used downstream.
"""

import numpy as np
import pandas as pd
from src.utils import (
    DATA_PATH,
    outcome_binary,
    ai_level_group,
    OUTCOME_SUCCESS,
)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read dataset.csv and return a raw DataFrame."""
    df = pd.read_csv(path, parse_dates=["SessionDate"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Steps
    -----
    1.  Remove duplicates on SessionID.
    2.  Coerce numeric columns; drop rows where critical fields are NaN.
    3.  Clamp out-of-range values (SatisfactionRating 1–5, AI_AssistanceLevel 1–5).
    4.  Remove extreme outliers in SessionLengthMin (IQR × 3 rule).
    5.  Engineer derived columns.

    Returns
    -------
    pd.DataFrame  –  Cleaned DataFrame.
    """
    df = df.copy()

    # ── 1. Deduplicate ──────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset="SessionID")
    n_dupes = before - len(df)
    if n_dupes:
        print(f"  [clean] Removed {n_dupes} duplicate SessionIDs.")

    # ── 2. Coerce numeric ───────────────────────────────────────────────────
    numeric_cols = ["SessionLengthMin", "TotalPrompts", "AI_AssistanceLevel", "SatisfactionRating"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_critical = df[numeric_cols + ["FinalOutcome", "StudentLevel", "TaskType"]].isnull().any(axis=1)
    n_miss = missing_critical.sum()
    if n_miss:
        print(f"  [clean] Dropping {n_miss} rows with missing critical fields.")
        df = df[~missing_critical].copy()

    # ── 3. Clamp ranges ─────────────────────────────────────────────────────
    df["SatisfactionRating"] = df["SatisfactionRating"].clip(1, 5)
    df["AI_AssistanceLevel"] = df["AI_AssistanceLevel"].clip(1, 5).astype(int)
    df["TotalPrompts"] = df["TotalPrompts"].clip(lower=1).astype(int)
    df["SessionLengthMin"] = df["SessionLengthMin"].clip(lower=0)

    # ── 4. Outlier removal – SessionLengthMin (IQR × 3) ────────────────────
    q1 = df["SessionLengthMin"].quantile(0.25)
    q3 = df["SessionLengthMin"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 3 * iqr
    n_out = (df["SessionLengthMin"] > upper_fence).sum()
    if n_out:
        print(f"  [clean] Capping {n_out} extreme SessionLengthMin values at {upper_fence:.1f} min.")
        df["SessionLengthMin"] = df["SessionLengthMin"].clip(upper=upper_fence)

    # ── 5. Derived columns ──────────────────────────────────────────────────
    df = _engineer_features(df)

    print(f"  [clean] Final dataset: {len(df):,} rows × {len(df.columns)} columns.")
    return df.reset_index(drop=True)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived / engineered columns to the cleaned DataFrame.

    New columns
    -----------
    success           : int   – 1 if task succeeded (Completed/Drafted), else 0
    ai_level_group    : str   – Low / Medium / High AI assistance tier
    prompts_per_min   : float – Prompt intensity = TotalPrompts / SessionLengthMin
    high_reliance     : bool  – AI_AssistanceLevel >= 4
    low_reliance      : bool  – AI_AssistanceLevel <= 2
    satisfaction_hi   : bool  – SatisfactionRating >= 4
    gave_up           : bool  – FinalOutcome == 'Gave Up'
    confused          : bool  – FinalOutcome == 'Confused'
    session_month     : int   – Calendar month (1–12) of the session
    session_year      : int   – Calendar year of the session
    """
    df["success"] = outcome_binary(df["FinalOutcome"])
    df["ai_level_group"] = df["AI_AssistanceLevel"].apply(ai_level_group)

    # Guard against zero-length sessions
    df["prompts_per_min"] = np.where(
        df["SessionLengthMin"] > 0,
        df["TotalPrompts"] / df["SessionLengthMin"],
        np.nan,
    )

    df["high_reliance"] = df["AI_AssistanceLevel"] >= 4
    df["low_reliance"] = df["AI_AssistanceLevel"] <= 2
    df["satisfaction_hi"] = df["SatisfactionRating"] >= 4
    df["gave_up"] = df["FinalOutcome"] == "Gave Up"
    df["confused"] = df["FinalOutcome"] == "Confused"

    if pd.api.types.is_datetime64_any_dtype(df["SessionDate"]):
        df["session_month"] = df["SessionDate"].dt.month
        df["session_year"] = df["SessionDate"].dt.year
    else:
        df["SessionDate"] = pd.to_datetime(df["SessionDate"], errors="coerce")
        df["session_month"] = df["SessionDate"].dt.month
        df["session_year"] = df["SessionDate"].dt.year

    return df


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def load_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """One-call convenience: load → clean → return."""
    print("[data_cleaning] Loading raw data …")
    raw = load_raw(path)
    print("[data_cleaning] Cleaning …")
    clean_df = clean(raw)
    print("[data_cleaning] Done.")
    return clean_df


if __name__ == "__main__":
    df = load_clean()
    print(df.head())
    print(df.dtypes)
