"""
visualization.py
----------------
Publication-quality chart generation for the AI Productivity Study.
All figures are saved to charts/ with 300 DPI.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from src.utils import CHARTS_DIR, ensure_dirs, ai_level_group, student_level_order, task_type_order

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

PALETTE = {
    "Low AI (1–2)":    "#4C72B0",
    "Medium AI (3)":   "#55A868",
    "High AI (4–5)":   "#C44E52",
    "High School":     "#8172B2",
    "Undergraduate":   "#CCB974",
    "Graduate":        "#64B5CD",
}

OUTCOME_COLORS = {
    "Assignment Completed": "#2E8B57",
    "Idea Drafted":         "#4682B4",
    "Confused":             "#DAA520",
    "Gave Up":              "#CD5C5C",
}

DPI = 300
CHART_EXT = ".png"


def _save(fig: plt.Figure, name: str) -> str:
    ensure_dirs()
    path = os.path.join(CHARTS_DIR, name + CHART_EXT)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] Saved → {path}")
    return path


# ============================================================================
# 1. productivity_comparison.png
#    Session length and prompts by AI assistance level
# ============================================================================

def plot_productivity_comparison(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "AI Assistance Level vs Session Efficiency",
        fontsize=16, fontweight="bold", y=1.01,
    )

    level_order = [1, 2, 3, 4, 5]
    colors = sns.color_palette("Blues_d", n_colors=5)

    # Panel A – Mean SessionLengthMin by AI level
    ax = axes[0]
    means = df.groupby("AI_AssistanceLevel")["SessionLengthMin"].mean()
    sems  = df.groupby("AI_AssistanceLevel")["SessionLengthMin"].sem()
    bars  = ax.bar(level_order, [means[l] for l in level_order],
                   yerr=[sems[l]*1.96 for l in level_order],
                   color=colors, edgecolor="white", linewidth=0.8,
                   error_kw=dict(ecolor="black", capsize=5, lw=1.5))
    ax.set_xlabel("AI Assistance Level", labelpad=8)
    ax.set_ylabel("Mean Session Length (min)")
    ax.set_title("A  Session Length by AI Assistance Level", loc="left", fontweight="bold")
    ax.set_xticks(level_order)
    ax.set_xticklabels(["1\n(Minimal)", "2\n(Low)", "3\n(Moderate)", "4\n(High)", "5\n(Full)"])
    for bar, mean in zip(bars, [means[l] for l in level_order]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B – TotalPrompts distribution by AI level (violin)
    ax = axes[1]
    data_list = [df.loc[df["AI_AssistanceLevel"] == lvl, "TotalPrompts"].dropna() for lvl in level_order]
    parts = ax.violinplot(data_list, positions=level_order, showmedians=True,
                          showextrema=False, widths=0.7)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.85)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xlabel("AI Assistance Level", labelpad=8)
    ax.set_ylabel("Total Prompts per Session")
    ax.set_title("B  Prompt Volume Distribution by AI Level", loc="left", fontweight="bold")
    ax.set_xticks(level_order)
    ax.set_xticklabels(["1\n(Minimal)", "2\n(Low)", "3\n(Moderate)", "4\n(High)", "5\n(Full)"])

    fig.tight_layout()
    return _save(fig, "productivity_comparison")


# ============================================================================
# 2. quality_scores.png
#    Satisfaction rating by task type and student level
# ============================================================================

def plot_quality_scores(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Output Quality: Satisfaction Ratings", fontsize=16, fontweight="bold", y=1.01)

    task_order = task_type_order()
    tasks_present = [t for t in task_order if t in df["TaskType"].unique()]

    # Panel A – Bar chart of mean satisfaction by TaskType
    ax = axes[0]
    task_means = df.groupby("TaskType")["SatisfactionRating"].mean().reindex(tasks_present)
    task_sems  = df.groupby("TaskType")["SatisfactionRating"].sem().reindex(tasks_present)
    colors_task = sns.color_palette("Set2", n_colors=len(tasks_present))
    bars = ax.barh(tasks_present, task_means, xerr=task_sems*1.96,
                   color=colors_task, edgecolor="white",
                   error_kw=dict(ecolor="gray", capsize=4, lw=1.2))
    ax.set_xlabel("Mean Satisfaction Rating (1–5)")
    ax.set_title("A  Mean Satisfaction by Task Type", loc="left", fontweight="bold")
    ax.axvline(df["SatisfactionRating"].mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"Overall mean = {df['SatisfactionRating'].mean():.2f}")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, task_means):
        ax.text(val + 0.04, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9, fontweight="bold")

    # Panel B – Box plot of satisfaction by student level
    ax = axes[1]
    level_order = student_level_order()
    levels_present = [l for l in level_order if l in df["StudentLevel"].unique()]
    palette_lvl = {l: PALETTE.get(l, "#999") for l in levels_present}
    sns.boxplot(data=df, x="StudentLevel", y="SatisfactionRating",
                order=levels_present, palette=palette_lvl, ax=ax,
                linewidth=1.2, fliersize=2, flierprops=dict(alpha=0.3))
    ax.set_xlabel("Student Level")
    ax.set_ylabel("Satisfaction Rating (1–5)")
    ax.set_title("B  Satisfaction by Student Level", loc="left", fontweight="bold")
    ax.axhline(df["SatisfactionRating"].mean(), color="black", linestyle="--",
               linewidth=1.2, label=f"Overall mean = {df['SatisfactionRating'].mean():.2f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, "quality_scores")


# ============================================================================
# 3. trust_vs_accuracy.png
#    UsedAgain rate and struggle rate by AI assistance level
# ============================================================================

def plot_trust_vs_accuracy(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Trust, Reliance, and Error Rate by AI Level",
                 fontsize=16, fontweight="bold", y=1.01)

    level_order = [1, 2, 3, 4, 5]
    df2 = df.copy()
    df2["struggle"] = df2["FinalOutcome"].isin({"Confused", "Gave Up"}).astype(int)

    used_again = df2.groupby("AI_AssistanceLevel")["UsedAgain"].mean() * 100
    struggle   = df2.groupby("AI_AssistanceLevel")["struggle"].mean() * 100

    ax = axes[0]
    ax.plot(level_order, [used_again[l] for l in level_order],
            marker="o", color="#4C72B0", linewidth=2.5, markersize=8, label="Would Use Again %")
    ax.plot(level_order, [struggle[l] for l in level_order],
            marker="s", color="#C44E52", linewidth=2.5, markersize=8, linestyle="--",
            label="Struggle Rate %")
    ax.fill_between(level_order, [used_again[l] for l in level_order], alpha=0.08, color="#4C72B0")
    ax.fill_between(level_order, [struggle[l] for l in level_order], alpha=0.08, color="#C44E52")
    ax.set_xlabel("AI Assistance Level")
    ax.set_ylabel("Rate (%)")
    ax.set_title("A  Trust & Struggle Rate by AI Level", loc="left", fontweight="bold")
    ax.set_xticks(level_order)
    ax.set_xticklabels(["1\n(Minimal)", "2\n(Low)", "3\n(Moderate)", "4\n(High)", "5\n(Full)"])
    ax.legend()

    # Panel B – Stacked bar of outcomes by AI level
    ax = axes[1]
    outcomes = ["Assignment Completed", "Idea Drafted", "Confused", "Gave Up"]
    outcome_pcts = (
        df2.groupby("AI_AssistanceLevel")["FinalOutcome"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=outcomes, fill_value=0)
        * 100
    )
    bottom = np.zeros(len(level_order))
    for outcome in outcomes:
        vals = [outcome_pcts.loc[l, outcome] if l in outcome_pcts.index else 0 for l in level_order]
        ax.bar(level_order, vals, bottom=bottom,
               color=OUTCOME_COLORS.get(outcome, "#aaa"), label=outcome, width=0.6)
        bottom += np.array(vals)
    ax.set_xlabel("AI Assistance Level")
    ax.set_ylabel("Proportion of Sessions (%)")
    ax.set_title("B  Outcome Distribution by AI Level", loc="left", fontweight="bold")
    ax.set_xticks(level_order)
    ax.set_xticklabels(["1\n(Minimal)", "2\n(Low)", "3\n(Moderate)", "4\n(High)", "5\n(Full)"])
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return _save(fig, "trust_vs_accuracy")


# ============================================================================
# 4. skill_retention.png
#    Gave Up / Confused rates as proxies for skill degradation at high AI levels
# ============================================================================

def plot_skill_retention(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Skill Retention Risk: Struggle Indicators by AI Reliance",
                 fontsize=16, fontweight="bold", y=1.01)

    ai_groups = ["Low AI (1–2)", "Medium AI (3)", "High AI (4–5)"]
    present   = [g for g in ai_groups if g in df["ai_level_group"].unique()]
    colors3   = ["#4C72B0", "#55A868", "#C44E52"]

    # Panel A – Confusion + Gave Up rates by AI group
    ax = axes[0]
    confused_r = df.groupby("ai_level_group")["confused"].mean().reindex(present) * 100
    gaveup_r   = df.groupby("ai_level_group")["gave_up"].mean().reindex(present) * 100
    x = np.arange(len(present))
    width = 0.35
    b1 = ax.bar(x - width/2, confused_r, width, label="Confused", color="#DAA520", edgecolor="white")
    b2 = ax.bar(x + width/2, gaveup_r, width, label="Gave Up", color="#CD5C5C", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=0)
    ax.set_ylabel("Rate (%)")
    ax.set_title("A  Confusion & Gave-Up Rate by AI Reliance", loc="left", fontweight="bold")
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8.5)

    # Panel B – Session length as function of prompts colored by outcome
    ax = axes[1]
    outcome_map = {"Assignment Completed": "#2E8B57", "Idea Drafted": "#4682B4",
                   "Confused": "#DAA520", "Gave Up": "#CD5C5C"}
    for outcome, color in outcome_map.items():
        sub = df[df["FinalOutcome"] == outcome]
        ax.scatter(sub["TotalPrompts"], sub["SessionLengthMin"], alpha=0.18,
                   color=color, s=12, label=outcome, rasterized=True)
    ax.set_xlabel("Total Prompts per Session")
    ax.set_ylabel("Session Length (min)")
    ax.set_title("B  Session Complexity vs Outcome", loc="left", fontweight="bold")
    ax.legend(markerscale=2, fontsize=9)

    fig.tight_layout()
    return _save(fig, "skill_retention")


# ============================================================================
# 5. task_category_effects.png
#    Success rate, satisfaction, and AI level distribution by TaskType
# ============================================================================

def plot_task_category_effects(df: pd.DataFrame) -> str:
    task_order = task_type_order()
    tasks_present = [t for t in task_order if t in df["TaskType"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Task Category Effects on AI-Assisted Performance",
                 fontsize=16, fontweight="bold", y=1.01)

    # Panel A – Success rate by task type, split by student level
    ax = axes[0]
    level_order = student_level_order()
    data_plot = (
        df.groupby(["TaskType", "StudentLevel"])["success"]
        .mean()
        .unstack()
        .reindex(index=tasks_present)
        .reindex(columns=level_order, fill_value=np.nan)
        * 100
    )
    x = np.arange(len(tasks_present))
    width = 0.25
    for i, lvl in enumerate(level_order):
        if lvl in data_plot.columns:
            ax.bar(x + i*width, data_plot[lvl], width, label=lvl,
                   color=PALETTE.get(lvl, "#aaa"), edgecolor="white")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks_present, rotation=30, ha="right")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("A  Success Rate by Task Type & Student Level", loc="left", fontweight="bold")
    ax.legend(title="Student Level")

    # Panel B – Heatmap: mean satisfaction by Task × AI level group
    ax = axes[1]
    heat_data = (
        df.groupby(["TaskType", "ai_level_group"])["SatisfactionRating"]
        .mean()
        .unstack()
        .reindex(index=tasks_present)
        .reindex(columns=["Low AI (1–2)", "Medium AI (3)", "High AI (4–5)"], fill_value=np.nan)
    )
    sns.heatmap(heat_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Mean Satisfaction"})
    ax.set_xlabel("AI Reliance Group")
    ax.set_ylabel("Task Type")
    ax.set_title("B  Satisfaction Heatmap (Task × AI Level)", loc="left", fontweight="bold")

    fig.tight_layout()
    return _save(fig, "task_category_effects")


# ============================================================================
# 6. confidence_gap.png
#    Satisfaction rating gap: completers vs. confused/gave-up
#    + correlation scatter of AI level vs satisfaction
# ============================================================================

def plot_confidence_gap(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confidence Gap: Perceived vs Actual Performance",
                 fontsize=16, fontweight="bold", y=1.01)

    # Panel A – Satisfaction distribution by FinalOutcome (ridge-style via violin)
    ax = axes[0]
    outcomes = ["Assignment Completed", "Idea Drafted", "Confused", "Gave Up"]
    outcomes_present = [o for o in outcomes if o in df["FinalOutcome"].unique()]
    data_list  = [df.loc[df["FinalOutcome"] == o, "SatisfactionRating"].dropna() for o in outcomes_present]
    palette_oc = [OUTCOME_COLORS.get(o, "#aaa") for o in outcomes_present]
    parts = ax.violinplot(data_list, positions=range(len(outcomes_present)),
                          showmedians=True, showextrema=False, widths=0.7)
    for pc, col in zip(parts["bodies"], palette_oc):
        pc.set_facecolor(col)
        pc.set_alpha(0.8)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2.5)
    means_oc = [d.mean() for d in data_list]
    ax.scatter(range(len(outcomes_present)), means_oc, color="black", zorder=5, s=50,
               label="Mean satisfaction")
    ax.set_xticks(range(len(outcomes_present)))
    ax.set_xticklabels(outcomes_present, rotation=15, ha="right")
    ax.set_ylabel("Satisfaction Rating (1–5)")
    ax.set_title("A  Satisfaction by Final Outcome", loc="left", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B – Scatter: AI level vs satisfaction with regression line
    ax = axes[1]
    jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(df))
    sc = ax.scatter(df["AI_AssistanceLevel"] + jitter, df["SatisfactionRating"],
                    alpha=0.08, s=8, color="#4C72B0", rasterized=True)
    # Regression line
    slope, intercept, r, p, se = stats.linregress(df["AI_AssistanceLevel"], df["SatisfactionRating"])
    x_line = np.linspace(1, 5, 100)
    ax.plot(x_line, intercept + slope * x_line, color="#C44E52", linewidth=2.5,
            label=f"Trend: r = {r:.3f}, p {'<.001' if p < 0.001 else f'= {p:.3f}'}")
    ax.set_xlabel("AI Assistance Level (jittered)")
    ax.set_ylabel("Satisfaction Rating (1–5)")
    ax.set_title("B  AI Level vs Satisfaction (Regression)", loc="left", fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=10)

    fig.tight_layout()
    return _save(fig, "confidence_gap")


# ============================================================================
# Master runner
# ============================================================================

def generate_all_charts(df: pd.DataFrame) -> dict:
    """Generate all 6 publication charts and return a dict of {name: path}."""
    ensure_dirs()
    paths = {}
    print("  Generating productivity_comparison …")
    paths["productivity_comparison"] = plot_productivity_comparison(df)
    print("  Generating quality_scores …")
    paths["quality_scores"] = plot_quality_scores(df)
    print("  Generating trust_vs_accuracy …")
    paths["trust_vs_accuracy"] = plot_trust_vs_accuracy(df)
    print("  Generating skill_retention …")
    paths["skill_retention"] = plot_skill_retention(df)
    print("  Generating task_category_effects …")
    paths["task_category_effects"] = plot_task_category_effects(df)
    print("  Generating confidence_gap …")
    paths["confidence_gap"] = plot_confidence_gap(df)
    return paths
