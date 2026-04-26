"""
build_notebook.py
-----------------
Generates analysis.ipynb programmatically.
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(PROJECT_ROOT, "analysis.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


CELLS = [
    md(
        "# How AI Changes Human Work\n"
        "## Productivity, Reliance, and Skill Retention\n\n"
        "**Empirical Analysis of 10,000 AI-Assisted Learning Sessions**  \n"
        "**Period:** June 2024 – June 2025 | **N = 10,000 sessions**\n\n"
        "---\n\n"
        "### Research Questions\n"
        "1. Does AI assistance level affect session efficiency?\n"
        "2. Does AI assistance predict task success?\n"
        "3. Does high AI reliance correlate with failure (overreliance)?\n"
        "4. Which task types benefit most from AI?\n"
        "5. Do beginners benefit more than experts?\n"
        "6. Does satisfaction correlate with AI level and outcomes?\n"
        "7. Can we predict struggle from session features?\n"
        "8. What predicts willingness to use AI again?\n"
    ),

    md("## 0. Setup & Imports"),

    code(
        "import sys\n"
        "import os\n"
        "import json\n"
        "import warnings\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy import stats\n"
        "import statsmodels.formula.api as smf\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.preprocessing import LabelEncoder, StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n\n"
        "warnings.filterwarnings('ignore')\n"
        "matplotlib.rcParams.update({\n"
        "    'font.family': 'DejaVu Sans',\n"
        "    'axes.spines.top': False,\n"
        "    'axes.spines.right': False,\n"
        "    'figure.dpi': 120,\n"
        "})\n\n"
        "sys.path.insert(0, os.getcwd())\n"
        "print('All imports successful.')\n"
    ),

    md("## 1. Data Loading & Cleaning\n\n"
       "Steps: duplicate detection, range clamping, IQR×3 outlier winsorization, feature engineering."),

    code(
        "from src.data_cleaning import load_clean\n\n"
        "df = load_clean('dataset.csv')\n"
        "print(f'Dataset shape: {df.shape}')\n"
        "df.head()\n"
    ),

    code("df.describe(include='all')\n"),

    code(
        "print('Missing values:')\n"
        "print(df.isnull().sum())\n"
        "print(f'Total missing: {df.isnull().sum().sum()}')\n"
    ),

    code(
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "df['AI_AssistanceLevel'].value_counts().sort_index().plot(\n"
        "    kind='bar', ax=axes[0], color='steelblue', edgecolor='white')\n"
        "axes[0].set_title('AI Assistance Level Distribution')\n"
        "df['FinalOutcome'].value_counts().plot(\n"
        "    kind='barh', ax=axes[1], color='coral', edgecolor='white')\n"
        "axes[1].set_title('Final Outcome Distribution')\n"
        "df['SatisfactionRating'].hist(ax=axes[2], bins=30,\n"
        "    color='mediumseagreen', edgecolor='white')\n"
        "axes[2].set_title('Satisfaction Rating Distribution')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),

    md("## 2. Exploratory Data Analysis"),

    code(
        "print('=== Student Level Counts ===')\n"
        "print(df['StudentLevel'].value_counts())\n"
        "print()\n"
        "print('=== Discipline Counts ===')\n"
        "print(df['Discipline'].value_counts())\n"
        "print()\n"
        "print('=== Task Type Counts ===')\n"
        "print(df['TaskType'].value_counts())\n"
        "print()\n"
        "print('=== Date Range ===')\n"
        "print(f\"First: {df['SessionDate'].min().date()}  |  Last: {df['SessionDate'].max().date()}\")\n"
    ),

    code(
        "# Session length and prompts by AI level\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "palette = sns.color_palette('Blues_d', 5)\n"
        "means = df.groupby('AI_AssistanceLevel')['SessionLengthMin'].mean()\n"
        "sems  = df.groupby('AI_AssistanceLevel')['SessionLengthMin'].sem()\n"
        "axes[0].bar(range(1,6), means, yerr=sems*1.96,\n"
        "            color=palette, capsize=5, edgecolor='white')\n"
        "axes[0].set_title('Mean Session Length by AI Level (+/-95% CI)')\n"
        "axes[0].set_xlabel('AI Assistance Level')\n"
        "axes[0].set_ylabel('Minutes')\n"
        "axes[0].set_xticks(range(1,6))\n\n"
        "df.boxplot(column='TotalPrompts', by='AI_AssistanceLevel', ax=axes[1],\n"
        "           patch_artist=True, medianprops=dict(color='white', lw=2))\n"
        "axes[1].set_title('Prompt Count by AI Level')\n"
        "axes[1].set_xlabel('AI Assistance Level')\n"
        "plt.suptitle('')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),

    code(
        "success_overall = df['success'].mean()\n"
        "print(f'Overall task success rate: {success_overall:.1%}')\n"
        "print(f\"Would use AI again: {df['UsedAgain'].mean():.1%}\")\n"
        "print()\n"
        "print('Success rate by AI level:')\n"
        "print(df.groupby('AI_AssistanceLevel')['success'].mean().round(3))\n"
        "print()\n"
        "print('Outcome breakdown:')\n"
        "print(df['FinalOutcome'].value_counts(normalize=True).round(3))\n"
    ),

    md("## 3. Statistical Analysis\n\n### RQ1: Session Efficiency\n\n"
       "**Hypothesis:** Higher AI reliance leads to shorter sessions."),

    code(
        "from src.statistical_analysis import rq1_session_length\n\n"
        "r1 = rq1_session_length(df)\n"
        "print('=== RQ1: Session Length vs AI Level ===')\n"
        "print('Group means (min):')\n"
        "for lvl, mean in r1['group_means_session_min'].items():\n"
        "    print(f'  Level {lvl}: {mean:.2f} min')\n"
        "print()\n"
        "print(f\"ANOVA: F={r1['anova_f']}, p={r1['anova_p']} {r1['anova_sig']}\")\n"
        "tt = r1['ttest_low_vs_high']\n"
        "print(f\"Low AI mean: {tt['mean_1']:.2f}  |  High AI mean: {tt['mean_2']:.2f}\")\n"
        "print(f\"Cohen's d = {tt['d']} ({tt['effect_label']})\")\n"
        "print()\n"
        "print('Interpretation:', r1['interpretation'])\n"
    ),

    md("### RQ2: Task Success"),

    code(
        "from src.statistical_analysis import rq2_task_success\n\n"
        "r2 = rq2_task_success(df)\n"
        "print(f\"Overall success rate: {r2['overall_success_rate']:.1%}\")\n"
        "print(f\"Chi-square: chi2({r2['dof']}) = {r2['chi2']}, p={r2['p_chi']}\")\n"
        "print(f\"Cramer's V = {r2['cramers_v']}\")\n"
        "print()\n"
        "print('Success rate by AI level:')\n"
        "for lvl, rate in r2['success_rate_by_level'].items():\n"
        "    print(f'  Level {lvl}: {rate:.1%}')\n"
        "if r2.get('logistic'):\n"
        "    lg = r2['logistic']\n"
        "    print(f\"OR (AI level): {lg['odds_ratio']}, p={lg['pval_AI_level']}\")\n"
        "print()\n"
        "print('Interpretation:', r2['interpretation'])\n"
    ),

    code(
        "# Stacked bar: outcomes by AI level\n"
        "pivot = df.groupby(['AI_AssistanceLevel', 'FinalOutcome']).size().unstack(fill_value=0)\n"
        "pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100\n"
        "ax = pivot_pct.plot(kind='bar', stacked=True, figsize=(10, 5),\n"
        "    color=['#2E8B57','#4682B4','#DAA520','#CD5C5C'], edgecolor='white')\n"
        "ax.set_title('Outcome Distribution by AI Assistance Level')\n"
        "ax.set_xlabel('AI Assistance Level')\n"
        "ax.set_ylabel('Percentage of Sessions (%)')\n"
        "ax.set_xticklabels(ax.get_xticklabels(), rotation=0)\n"
        "ax.legend(loc='upper right', title='Final Outcome')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),

    md("### RQ3: Overreliance"),

    code(
        "from src.statistical_analysis import rq3_overreliance\n\n"
        "r3 = rq3_overreliance(df)\n"
        "print(f\"Struggle rate -- High AI: {r3['struggle_rate_high_AI']:.1%}\")\n"
        "print(f\"Struggle rate -- Low AI:  {r3['struggle_rate_low_AI']:.1%}\")\n"
        "print(f\"Chi-square: chi2({r3['dof']}) = {r3['chi2']}, p={r3['p']}\")\n"
        "print()\n"
        "print('Gave-Up rate by level:')\n"
        "for lvl, rate in r3['gaveup_by_level'].items():\n"
        "    print(f'  Level {lvl}: {rate:.1%}')\n"
        "print()\n"
        "print('Interpretation:', r3['interpretation'])\n"
    ),

    md("### RQ4: Task Type Effects"),

    code(
        "from src.statistical_analysis import rq4_task_type_effects\n\n"
        "r4 = rq4_task_type_effects(df)\n"
        "print(f\"ANOVA: F={r4['anova_f']}, p={r4['anova_p']} {r4['anova_sig']}\")\n"
        "print()\n"
        "sat_data = r4['satisfaction_by_task']\n"
        "for task, data in r4['success_by_task'].items():\n"
        "    sat = sat_data['mean_satisfaction'].get(task, float('nan'))\n"
        "    print(f\"  {task:<20} success={data['success_rate']:.1%}  sat={sat:.2f}\")\n"
    ),

    code(
        "heat = (\n"
        "    df.groupby(['TaskType', 'ai_level_group'])['SatisfactionRating']\n"
        "    .mean().unstack()\n"
        ")\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "sns.heatmap(heat, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,\n"
        "            linewidths=0.5, cbar_kws={'label': 'Mean Satisfaction'})\n"
        "ax.set_title('Satisfaction by Task Type x AI Level Group')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),

    md("### RQ5: Beginners vs Experts"),

    code(
        "from src.statistical_analysis import rq5_beginner_vs_expert\n\n"
        "r5 = rq5_beginner_vs_expert(df)\n"
        "print('Success rates:')\n"
        "for lvl, rate in r5['success_by_level'].items():\n"
        "    print(f'  {lvl}: {rate:.1%}')\n"
        "tt = r5['success_ttest_hs_vs_grad']\n"
        "print(f\"\\nHS vs Grad: t={tt['t']}, p={tt['p']}, Cohen's d={tt['d']} ({tt['effect_label']})\")\n"
        "print('\\nInterpretation:', r5['interpretation'])\n"
    ),

    md("### RQ6: The Satisfaction Inflation Effect\n\n"
       "> **Key finding:** AI level is strongly correlated with satisfaction (r=0.776, p<.001, R2=0.60) "
       "but satisfaction shows NO significant relationship with actual task success. "
       "This decoupling is the study's central contribution."),

    code(
        "from src.statistical_analysis import rq6_satisfaction_correlates\n\n"
        "r6 = rq6_satisfaction_correlates(df)\n"
        "print(f\"r(Satisfaction, AI Level) = {r6['r_sat_ai']:.3f},  p = {r6['p_sat_ai']}\")\n"
        "print(f\"r(Satisfaction, Success)  = {r6['r_sat_success']:.3f},  p = {r6['p_sat_success']}\")\n"
        "print(f\"r(Satisfaction, Prompts)  = {r6['r_sat_prompts']:.3f},  p = {r6['p_sat_prompts']}\")\n"
        "print()\n"
        "ols = r6.get('ols', {})\n"
        "if ols:\n"
        "    print(f\"OLS R2 = {ols['r_squared']}, adj-R2 = {ols['adj_r_squared']}\")\n"
        "    print('Coefficients:')\n"
        "    for param, coef in ols['params'].items():\n"
        "        pval = ols['pvalues'][param]\n"
        "        print(f'  {param:<30} coef={coef:>8.4f}  p={pval:.4f}')\n"
        "print()\n"
        "print('INTERPRETATION:', r6['interpretation'])\n"
    ),

    code(
        "# KEY CHART: Satisfaction Inflation Effect\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(df))\n"
        "axes[0].scatter(df['AI_AssistanceLevel'] + jitter, df['SatisfactionRating'],\n"
        "                alpha=0.06, s=8, color='#4C72B0', rasterized=True)\n"
        "slope, intercept, r, p, _ = stats.linregress(\n"
        "    df['AI_AssistanceLevel'], df['SatisfactionRating'])\n"
        "x_line = np.linspace(1, 5, 100)\n"
        "axes[0].plot(x_line, intercept + slope * x_line, color='#C44E52', lw=2.5,\n"
        "             label=f'r = {r:.3f} (p < .001)')\n"
        "axes[0].set_title('AI Assistance Level vs Satisfaction')\n"
        "axes[0].set_xlabel('AI Assistance Level (jittered)')\n"
        "axes[0].set_ylabel('Satisfaction Rating (1-5)')\n"
        "axes[0].set_xticks(range(1, 6))\n"
        "axes[0].legend()\n\n"
        "succ_means = df.groupby('success')['SatisfactionRating'].mean()\n"
        "axes[1].bar(['Struggle (0)', 'Success (1)'], succ_means,\n"
        "            color=['#CD5C5C', '#2E8B57'], edgecolor='white')\n"
        "axes[1].set_title('Mean Satisfaction by Task Success')\n"
        "axes[1].set_ylabel('Mean Satisfaction')\n"
        "for i, v in enumerate(succ_means):\n"
        "    axes[1].text(i, v + 0.04, f'{v:.3f}', ha='center', fontweight='bold')\n"
        "plt.suptitle('The Satisfaction Inflation Effect', fontsize=14, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),

    md("### RQ7: Predictive Model"),

    code(
        "from src.statistical_analysis import rq7_predictive_model\n\n"
        "r7 = rq7_predictive_model(df)\n"
        "print(f\"5-fold CV AUC = {r7['cv_roc_auc_mean']:.3f} (SD={r7['cv_roc_auc_std']:.3f})\")\n"
        "print()\n"
        "print('Feature coefficients (standardized):')\n"
        "for feat, coef in r7['coefficients'].items():\n"
        "    print(f'  {feat:<25} {coef:+.4f}')\n"
        "print()\n"
        "print('Note: AUC near 0.5 -- session features are weak predictors of individual struggle.')\n"
        "print('Interpretation:', r7['interpretation'])\n"
    ),

    md("### RQ8: Repeat Usage"),

    code(
        "from src.statistical_analysis import rq8_repeat_usage\n\n"
        "r8 = rq8_repeat_usage(df)\n"
        "print(f\"UsedAgain rate: {r8['used_again_rate']:.1%}\")\n"
        "print(f\"r(UsedAgain, Satisfaction): {r8['r_used_satisfaction']:.3f}  p={r8['p_used_satisfaction']}\")\n"
        "print(f\"r(UsedAgain, Success):      {r8['r_used_success']:.3f}  p={r8['p_used_success']}\")\n"
        "print()\n"
        "print('UsedAgain rate by outcome:')\n"
        "for outcome, rate in r8['used_again_by_outcome'].items():\n"
        "    print(f'  {outcome:<25}: {rate:.1%}')\n"
        "print('\\nInterpretation:', r8['interpretation'])\n"
    ),

    md("## 4. Publication Charts"),

    code(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from src.visualization import generate_all_charts\n\n"
        "paths = generate_all_charts(df)\n"
        "print('Generated charts:')\n"
        "for name, path in paths.items():\n"
        "    print(f'  {name}: {path}')\n"
    ),

    md(
        "## 5. Summary of Findings\n\n"
        "| RQ | Finding | Effect | Significant? |\n"
        "|----|---------|--------|--------------|\n"
        "| RQ1: Session efficiency | No time difference by AI level | d = 0.015 | No |\n"
        "| RQ2: Task success | 76.3% overall; AI level not predictive | V = 0.019 | No |\n"
        "| RQ3: Overreliance | Struggle equal across AI groups (23%) | negligible | No |\n"
        "| RQ4: Task type | No satisfaction variation by task type | F = 1.05 | No |\n"
        "| RQ5: Beginner vs expert | Trivial performance gap (d=0.021) | negligible | No |\n"
        "| **RQ6: Satisfaction inflation** | **AI level strongly predicts satisfaction r=0.776** | **R2=0.60** | **Yes ***  |\n"
        "| RQ7: Struggle prediction | CV AUC = 0.534 (near-chance) | -- | Weak |\n"
        "| **RQ8: Repeat usage** | **Success predicts UsedAgain r=0.371** | -- | **Yes ***  |\n\n"
        "### Central Insight\n"
        "Students who use AI more heavily report substantially higher session satisfaction, "
        "but this satisfaction is **decoupled from actual task success**. This satisfaction inflation "
        "effect is a hallmark pattern of potential overreliance risk."
    ),

    code(
        "print('=' * 55)\n"
        "print('  STUDY SUMMARY STATISTICS')\n"
        "print('=' * 55)\n"
        "print(f'  Total sessions analyzed:    {len(df):,}')\n"
        "print(f\"  Date range:                 {df['SessionDate'].min().date()} to {df['SessionDate'].max().date()}\")\n"
        "print(f\"  Student levels:             {df['StudentLevel'].nunique()} (HS, UG, Grad)\")\n"
        "print(f\"  Disciplines:                {df['Discipline'].nunique()}\")\n"
        "print(f\"  Task types:                 {df['TaskType'].nunique()}\")\n"
        "print(f\"  Overall success rate:       {df['success'].mean():.1%}\")\n"
        "print(f\"  Would use AI again:         {df['UsedAgain'].mean():.1%}\")\n"
        "print(f\"  Mean session length:        {df['SessionLengthMin'].mean():.1f} min\")\n"
        "print(f\"  Mean prompts/session:       {df['TotalPrompts'].mean():.1f}\")\n"
        "print(f\"  Mean AI assistance level:   {df['AI_AssistanceLevel'].mean():.2f} / 5\")\n"
        "print(f\"  Mean satisfaction:          {df['SatisfactionRating'].mean():.2f} / 5\")\n"
        "print()\n"
        "print('  KEY RESULTS:')\n"
        "print('  r(Satisfaction ~ AI Level):   0.776 (p < .001)')\n"
        "print('  r(UsedAgain ~ Success):       0.371 (p < .001)')\n"
        "print('  OLS R-squared (Sat model):    0.602')\n"
        "print('  Logistic CV AUC (Struggle):   0.534')\n"
        "print('=' * 55)\n"
    ),
]


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": CELLS,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Written -> {NB_PATH}")


if __name__ == "__main__":
    pass
