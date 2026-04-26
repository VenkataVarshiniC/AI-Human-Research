"""
statistical_analysis.py
-----------------------
All inferential statistics for the AI Productivity Study.

Functions return structured dict objects suitable for embedding
in the Jupyter notebook and for PDF report generation.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from src.utils import welch_ttest, cohens_d, cramers_v, bootstrap_ci, fmt_sig


# ============================================================================
# RQ1 – Does AI assistance level affect session length (proxy for efficiency)?
# ============================================================================

def rq1_session_length(df: pd.DataFrame) -> dict:
    """
    Compare SessionLengthMin across AI assistance level groups.
    High reliance (4–5) vs Low reliance (1–2) — Welch's t-test + Cohen's d.
    Also run one-way ANOVA across all 5 levels.
    """
    low = df.loc[df["AI_AssistanceLevel"] <= 2, "SessionLengthMin"].dropna()
    high = df.loc[df["AI_AssistanceLevel"] >= 4, "SessionLengthMin"].dropna()
    ttest = welch_ttest(low, high)

    groups = [df.loc[df["AI_AssistanceLevel"] == lvl, "SessionLengthMin"].dropna() for lvl in range(1, 6)]
    f_stat, p_anova = stats.f_oneway(*groups)

    group_means = df.groupby("AI_AssistanceLevel")["SessionLengthMin"].mean().round(2).to_dict()

    # Prompt intensity analysis
    low_ppm = df.loc[df["AI_AssistanceLevel"] <= 2, "prompts_per_min"].dropna()
    high_ppm = df.loc[df["AI_AssistanceLevel"] >= 4, "prompts_per_min"].dropna()
    ppm_ttest = welch_ttest(low_ppm, high_ppm)

    return {
        "ttest_low_vs_high": ttest,
        "anova_f": round(f_stat, 4),
        "anova_p": round(p_anova, 6),
        "anova_sig": fmt_sig(p_anova),
        "group_means_session_min": group_means,
        "ppm_ttest": ppm_ttest,
        "interpretation": (
            f"Sessions with high AI reliance (levels 4–5) averaged "
            f"{ttest['mean_2']:.1f} min vs {ttest['mean_1']:.1f} min for low-reliance (1–2). "
            f"One-way ANOVA: F = {f_stat:.2f}, p {fmt_sig(p_anova)} "
            f"({'significant' if p_anova < 0.05 else 'not significant'} at α=0.05). "
            f"Cohen's d = {ttest['d']:.3f} ({ttest['effect_label']} effect)."
        ),
    }


# ============================================================================
# RQ2 – Does AI assistance level predict task success (outcome quality)?
# ============================================================================

def rq2_task_success(df: pd.DataFrame) -> dict:
    """
    Chi-square test: AI_AssistanceLevel group × FinalOutcome.
    Logistic regression: success ~ AI_AssistanceLevel.
    Success rate by level.
    """
    ct = pd.crosstab(df["ai_level_group"], df["FinalOutcome"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    v = cramers_v(ct)

    success_rate = df.groupby("AI_AssistanceLevel")["success"].mean().round(3).to_dict()
    overall_success = df["success"].mean()

    # Logistic regression
    sub = df[["AI_AssistanceLevel", "success"]].dropna()
    X = sm.add_constant(sub["AI_AssistanceLevel"])
    y = sub["success"]
    try:
        logit = sm.Logit(y, X).fit(disp=False)
        logit_summary = {
            "coef_AI_level": round(logit.params["AI_AssistanceLevel"], 4),
            "pval_AI_level": round(logit.pvalues["AI_AssistanceLevel"], 6),
            "odds_ratio": round(np.exp(logit.params["AI_AssistanceLevel"]), 4),
            "pseudo_r2": round(logit.prsquared, 4),
        }
    except Exception:
        logit_summary = {}

    return {
        "chi2": round(chi2, 4),
        "p_chi": round(p_chi, 6),
        "dof": dof,
        "cramers_v": round(v, 4),
        "success_rate_by_level": success_rate,
        "overall_success_rate": round(overall_success, 4),
        "logistic": logit_summary,
        "interpretation": (
            f"Chi-square test (AI group × outcome): χ²({dof}) = {chi2:.2f}, "
            f"p {fmt_sig(p_chi)}. Cramér's V = {v:.3f}. "
            f"Overall task success rate = {overall_success:.1%}."
        ),
    }


# ============================================================================
# RQ3 – Does high AI reliance correlate with negative outcomes (Confused / Gave Up)?
# ============================================================================

def rq3_overreliance(df: pd.DataFrame) -> dict:
    """
    Overreliance defined as high_reliance AND (Confused OR Gave Up).
    Compute prevalence, compare to low-reliance group.
    """
    df = df.copy()
    df["struggle"] = df["FinalOutcome"].isin({"Confused", "Gave Up"}).astype(int)

    high = df[df["high_reliance"]]["struggle"]
    low  = df[df["low_reliance"]]["struggle"]

    struggle_rate_high = high.mean()
    struggle_rate_low  = low.mean()

    ct = pd.crosstab(df["high_reliance"], df["struggle"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)

    # Overreliance error rate = fraction of high-AI sessions that failed
    overreliance_rate = df[df["high_reliance"]]["gave_up"].mean()

    # Confusion rate by AI level
    confusion_by_level = df.groupby("AI_AssistanceLevel")["confused"].mean().round(3).to_dict()
    gaveup_by_level    = df.groupby("AI_AssistanceLevel")["gave_up"].mean().round(3).to_dict()

    return {
        "struggle_rate_high_AI": round(struggle_rate_high, 4),
        "struggle_rate_low_AI": round(struggle_rate_low, 4),
        "chi2": round(chi2, 4),
        "p": round(p, 6),
        "dof": dof,
        "overreliance_gave_up_rate": round(overreliance_rate, 4),
        "confusion_by_level": confusion_by_level,
        "gaveup_by_level": gaveup_by_level,
        "interpretation": (
            f"Struggle rate (Confused + Gave Up) for high-AI sessions: {struggle_rate_high:.1%} "
            f"vs low-AI sessions: {struggle_rate_low:.1%}. "
            f"Chi-square: χ²({dof}) = {chi2:.2f}, p {fmt_sig(p)}."
        ),
    }


# ============================================================================
# RQ4 – Which task types benefit most from AI?
# ============================================================================

def rq4_task_type_effects(df: pd.DataFrame) -> dict:
    """
    Success rate and satisfaction by TaskType.
    ANOVA on SatisfactionRating × TaskType.
    """
    success_by_task = df.groupby("TaskType")["success"].agg(["mean", "count"]).round(3)
    success_by_task.columns = ["success_rate", "n"]
    success_by_task = success_by_task.to_dict(orient="index")

    sat_by_task = df.groupby("TaskType")["SatisfactionRating"].agg(["mean", "std"]).round(3)
    sat_by_task.columns = ["mean_satisfaction", "std_satisfaction"]

    task_groups = [df.loc[df["TaskType"] == t, "SatisfactionRating"].dropna()
                   for t in df["TaskType"].unique()]
    f_stat, p_anova = stats.f_oneway(*task_groups)

    # Pairwise Tukey HSD via statsmodels
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(
            df["SatisfactionRating"].dropna(),
            df.loc[df["SatisfactionRating"].notna(), "TaskType"],
        )
        tukey_summary = str(tukey.summary())
    except Exception:
        tukey_summary = "Not available."

    return {
        "success_by_task": success_by_task,
        "satisfaction_by_task": sat_by_task.to_dict(),
        "anova_f": round(f_stat, 4),
        "anova_p": round(p_anova, 6),
        "anova_sig": fmt_sig(p_anova),
        "tukey_summary": tukey_summary,
        "interpretation": (
            f"ANOVA (SatisfactionRating × TaskType): F = {f_stat:.2f}, p {fmt_sig(p_anova)}."
        ),
    }


# ============================================================================
# RQ5 – Do beginners benefit more than experts?
# ============================================================================

def rq5_beginner_vs_expert(df: pd.DataFrame) -> dict:
    """
    Compare success rate, satisfaction, and session length across StudentLevel.
    Particular focus on High School (novice) vs Graduate (expert).
    """
    hs = df[df["StudentLevel"] == "High School"]
    grad = df[df["StudentLevel"] == "Graduate"]

    success_ttest = welch_ttest(hs["success"], grad["success"])
    sat_ttest     = welch_ttest(hs["SatisfactionRating"], grad["SatisfactionRating"])
    len_ttest     = welch_ttest(hs["SessionLengthMin"], grad["SessionLengthMin"])

    success_by_level = df.groupby("StudentLevel")["success"].mean().round(3).to_dict()
    sat_by_level     = df.groupby("StudentLevel")["SatisfactionRating"].mean().round(3).to_dict()

    # Interaction: AI level × student level → success rate
    interaction = (
        df.groupby(["StudentLevel", "ai_level_group"])["success"]
        .mean()
        .round(3)
        .unstack()
        .to_dict()
    )

    return {
        "success_ttest_hs_vs_grad": success_ttest,
        "sat_ttest_hs_vs_grad": sat_ttest,
        "len_ttest_hs_vs_grad": len_ttest,
        "success_by_level": success_by_level,
        "sat_by_level": sat_by_level,
        "interaction": interaction,
        "interpretation": (
            f"High School success rate: {success_ttest['mean_1']:.1%} | "
            f"Graduate success rate: {success_ttest['mean_2']:.1%}. "
            f"Cohen's d = {success_ttest['d']:.3f} ({success_ttest['effect_label']} effect)."
        ),
    }


# ============================================================================
# RQ6 – Does satisfaction correlate with outcome and AI level?
# ============================================================================

def rq6_satisfaction_correlates(df: pd.DataFrame) -> dict:
    """
    Pearson correlation of SatisfactionRating with AI_AssistanceLevel and success.
    Multivariate OLS: SatisfactionRating ~ AI_AssistanceLevel + success + TotalPrompts + SessionLengthMin.
    """
    r_ai, p_ai  = stats.pearsonr(df["SatisfactionRating"], df["AI_AssistanceLevel"])
    r_suc, p_suc = stats.pearsonr(df["SatisfactionRating"], df["success"])
    r_prom, p_prom = stats.pearsonr(df["SatisfactionRating"], df["TotalPrompts"])

    # Multivariate OLS
    formula = "SatisfactionRating ~ AI_AssistanceLevel + success + TotalPrompts + SessionLengthMin"
    try:
        ols = smf.ols(formula, data=df).fit()
        ols_summary = {
            "r_squared": round(ols.rsquared, 4),
            "adj_r_squared": round(ols.rsquared_adj, 4),
            "f_stat": round(ols.fvalue, 4),
            "f_p": round(ols.f_pvalue, 6),
            "params": {k: round(v, 4) for k, v in ols.params.items()},
            "pvalues": {k: round(v, 6) for k, v in ols.pvalues.items()},
        }
    except Exception as e:
        ols_summary = {"error": str(e)}

    return {
        "r_sat_ai": round(r_ai, 4),
        "p_sat_ai": round(p_ai, 6),
        "r_sat_success": round(r_suc, 4),
        "p_sat_success": round(p_suc, 6),
        "r_sat_prompts": round(r_prom, 4),
        "p_sat_prompts": round(p_prom, 6),
        "ols": ols_summary,
        "interpretation": (
            f"Satisfaction ↔ AI Level: r = {r_ai:.3f} (p {fmt_sig(p_ai)}). "
            f"Satisfaction ↔ Success: r = {r_suc:.3f} (p {fmt_sig(p_suc)}). "
            f"OLS R² = {ols_summary.get('r_squared', 'N/A')}."
        ),
    }


# ============================================================================
# RQ7 – Predictive model: overtrust / struggle likelihood
# ============================================================================

def rq7_predictive_model(df: pd.DataFrame) -> dict:
    """
    Logistic regression to predict 'struggle' (Confused or Gave Up)
    from AI_AssistanceLevel, TotalPrompts, SessionLengthMin, StudentLevel, TaskType.
    """
    model_df = df[["struggle" if "struggle" in df.columns else "success",
                   "AI_AssistanceLevel", "TotalPrompts", "SessionLengthMin",
                   "StudentLevel", "TaskType"]].copy()
    model_df["struggle"] = df["FinalOutcome"].isin({"Confused", "Gave Up"}).astype(int)

    le_student = LabelEncoder()
    le_task    = LabelEncoder()
    model_df["student_enc"] = le_student.fit_transform(model_df["StudentLevel"])
    model_df["task_enc"]    = le_task.fit_transform(model_df["TaskType"])

    features = ["AI_AssistanceLevel", "TotalPrompts", "SessionLengthMin",
                "student_enc", "task_enc"]
    X = model_df[features].dropna()
    y = model_df.loc[X.index, "struggle"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X_sc, y, cv=5, scoring="roc_auc")
    clf.fit(X_sc, y)

    coef_dict = dict(zip(features, clf.coef_[0].round(4)))

    return {
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "coefficients": coef_dict,
        "feature_names": features,
        "interpretation": (
            f"Logistic model predicting struggle: mean CV AUC = {cv_scores.mean():.3f} "
            f"(±{cv_scores.std():.3f}). "
            f"AI_AssistanceLevel coefficient = {coef_dict['AI_AssistanceLevel']:.3f}."
        ),
    }


# ============================================================================
# RQ8 – Repeat usage analysis (UsedAgain ~ satisfaction + success + AI level)
# ============================================================================

def rq8_repeat_usage(df: pd.DataFrame) -> dict:
    """
    Model willingness to use AI again.
    Binary outcome: UsedAgain.
    """
    sub = df[["UsedAgain", "SatisfactionRating", "success", "AI_AssistanceLevel",
              "TotalPrompts", "SessionLengthMin"]].copy()
    sub["used_again_int"] = sub["UsedAgain"].astype(int)

    r_sat, p_sat = stats.pointbiserialr(sub["used_again_int"], sub["SatisfactionRating"])
    r_suc, p_suc = stats.pointbiserialr(sub["used_again_int"], sub["success"])
    r_ai,  p_ai  = stats.pointbiserialr(sub["used_again_int"], sub["AI_AssistanceLevel"])

    used_again_rate   = sub["UsedAgain"].mean()
    used_by_outcome   = df.groupby("FinalOutcome")["UsedAgain"].mean().round(3).to_dict()
    used_by_ai_level  = df.groupby("AI_AssistanceLevel")["UsedAgain"].mean().round(3).to_dict()

    return {
        "used_again_rate": round(used_again_rate, 4),
        "r_used_satisfaction": round(r_sat, 4),
        "p_used_satisfaction": round(p_sat, 6),
        "r_used_success": round(r_suc, 4),
        "p_used_success": round(p_suc, 6),
        "r_used_ai_level": round(r_ai, 4),
        "p_used_ai_level": round(p_ai, 6),
        "used_again_by_outcome": used_by_outcome,
        "used_again_by_ai_level": used_by_ai_level,
        "interpretation": (
            f"{used_again_rate:.1%} of participants would use AI again. "
            f"Satisfaction ↔ UsedAgain: r = {r_sat:.3f} (p {fmt_sig(p_sat)}). "
            f"Success ↔ UsedAgain: r = {r_suc:.3f} (p {fmt_sig(p_suc)})."
        ),
    }


# ============================================================================
# Convenience runner
# ============================================================================

def run_all(df: pd.DataFrame) -> dict:
    """Run all research questions and return a results dictionary."""
    print("  Running RQ1: Session length & efficiency …")
    results = {"rq1": rq1_session_length(df)}
    print("  Running RQ2: Task success …")
    results["rq2"] = rq2_task_success(df)
    print("  Running RQ3: Overreliance …")
    results["rq3"] = rq3_overreliance(df)
    print("  Running RQ4: Task type effects …")
    results["rq4"] = rq4_task_type_effects(df)
    print("  Running RQ5: Beginner vs expert …")
    results["rq5"] = rq5_beginner_vs_expert(df)
    print("  Running RQ6: Satisfaction correlates …")
    results["rq6"] = rq6_satisfaction_correlates(df)
    print("  Running RQ7: Predictive model …")
    results["rq7"] = rq7_predictive_model(df)
    print("  Running RQ8: Repeat usage …")
    results["rq8"] = rq8_repeat_usage(df)
    return results
