"""
utils.py
--------
Shared utility functions for the AI Productivity Study analysis pipeline.
Provides helpers for statistical computations, formatting, and I/O.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset.csv")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")


def ensure_dirs() -> None:
    """Create required output directories if they don't exist."""
    os.makedirs(CHARTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent samples.

    Parameters
    ----------
    group1, group2 : array-like
        Numeric samples to compare.

    Returns
    -------
    float
        Cohen's d (positive means group1 > group2).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Return a plain-English label for an effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> dict:
    """
    Run Welch's independent-samples t-test and return a summary dict.

    Returns
    -------
    dict with keys: t, p, df, d, effect_label, significant
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    d = cohens_d(g1, g2)
    return {
        "t": round(t_stat, 4),
        "p": round(p_val, 6),
        "mean_1": round(g1.mean(), 4),
        "mean_2": round(g2.mean(), 4),
        "n1": len(g1),
        "n2": len(g2),
        "d": round(d, 4),
        "effect_label": interpret_cohens_d(d),
        "significant": p_val < 0.05,
    }


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Compute Cramér's V effect size from a contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Cross-tabulation of two categorical variables.

    Returns
    -------
    float
        Cramér's V in [0, 1].
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    k = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """
    Compute a bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data      : array-like  –  Input data.
    statistic : callable    –  Function to apply to each bootstrap sample.
    n_boot    : int         –  Number of bootstrap iterations.
    ci        : float       –  Confidence level (0–1).

    Returns
    -------
    (lower, upper) tuple of floats.
    """
    rng = np.random.default_rng(42)
    arr = np.asarray(data, dtype=float)
    boots = [statistic(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (np.percentile(boots, 100 * alpha), np.percentile(boots, 100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

AI_LEVEL_LABELS = {
    1: "Minimal (1)",
    2: "Low (2)",
    3: "Moderate (3)",
    4: "High (4)",
    5: "Full (5)",
}

OUTCOME_SUCCESS = {"Assignment Completed", "Idea Drafted"}
OUTCOME_FAILURE = {"Confused", "Gave Up"}


def outcome_binary(outcome_series: pd.Series) -> pd.Series:
    """Map FinalOutcome to 1 (success) or 0 (failure/struggle)."""
    return outcome_series.map(lambda x: 1 if x in OUTCOME_SUCCESS else 0)


def ai_level_group(level: int) -> str:
    """Collapse AI assistance level into Low / Medium / High."""
    if level <= 2:
        return "Low AI (1–2)"
    elif level == 3:
        return "Medium AI (3)"
    else:
        return "High AI (4–5)"


def student_level_order() -> list:
    return ["High School", "Undergraduate", "Graduate"]


def task_type_order() -> list:
    return ["Writing", "Studying", "Homework Help", "Coding", "Brainstorming", "Research"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a ratio [0,1] as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_sig(p: float) -> str:
    """Return p-value significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def summarize_group(series: pd.Series) -> dict:
    """Return mean ± std and 95% CI for a numeric series."""
    arr = series.dropna().values
    lo, hi = bootstrap_ci(arr)
    return {
        "n": len(arr),
        "mean": round(arr.mean(), 4),
        "std": round(arr.std(ddof=1), 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
    }
