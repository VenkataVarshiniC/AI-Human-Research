# How AI Changes Human Work
### Productivity, Reliance, and Skill Retention

Venkata Varshini Chilukamarri | Independent AI Researcher | University of Maryland

Empirical analysis of productivity, trust calibration, and skill retention across 10,000 AI-assisted work sessions.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-10%2C000_Sessions-orange.svg)](dataset.csv)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](analysis.ipynb)

Statistical Methods: Regression • ANOVA • Effect Sizes • Behavioral Metrics

---

## Problem Statement

The rapid adoption of AI assistants in academic and professional settings has outpaced our empirical understanding of *how* they change human performance. While AI tools demonstrably reduce friction for many tasks, open questions remain: Does AI assistance actually improve output quality, or merely reduce perceived effort? Does heavy reliance on AI erode independent skills over time? When does user trust tip into dangerous overreliance?

This research project addresses these questions using a real-world observational dataset of **10,000 AI-assisted learning sessions** collected from students across High School, Undergraduate, and Graduate levels spanning six task categories over 12 months.

---

## Why This Matters

- **For learners:** Understanding when AI helps vs. hinders retention can inform smarter AI usage habits.
- **For educators:** Empirical data on AI reliance patterns can guide policy on appropriate AI tool use.
- **For AI developers:** Identifying overreliance signatures helps design systems with appropriate guardrails.
- **For society:** The question of whether AI substitutes for or complements human capability has major implications for the future of knowledge work.

---

## Research Questions

| # | Question | Method |
|---|---|---|
| RQ1 | Does AI assistance level affect task efficiency (session time)? | Welch's t-test, One-way ANOVA |
| RQ2 | Does AI assistance predict task success? | Chi-square, Logistic regression |
| RQ3 | Does high AI reliance correlate with failure (overreliance)? | Chi-square, proportion testing |
| RQ4 | Which task types benefit most from AI assistance? | ANOVA, Tukey HSD, Heatmap |
| RQ5 | Do beginners benefit more than experts from AI? | Welch's t-test, interaction |
| RQ6 | Does satisfaction correlate with AI level and outcomes? | Pearson r, OLS regression |
| RQ7 | Can we predict overtrust/struggle from session features? | Logistic regression, 5-fold CV AUC |
| RQ8 | What predicts willingness to use AI again? | Point-biserial correlation |

---

## Dataset Description

**File:** `dataset.csv` | **Rows:** 10,000 | **Columns:** 11 (raw) + 10 engineered  
**Period:** June 24, 2024 – June 24, 2025

| Column | Type | Description |
|---|---|---|
| `SessionID` | string | Unique session identifier |
| `StudentLevel` | categorical | High School / Undergraduate / Graduate |
| `Discipline` | categorical | 7 academic disciplines |
| `SessionDate` | date | Date of session (YYYY-MM-DD) |
| `SessionLengthMin` | float | Session duration in minutes |
| `TotalPrompts` | int | Number of AI prompts submitted |
| `TaskType` | categorical | Writing / Studying / Homework Help / Coding / Brainstorming / Research |
| `AI_AssistanceLevel` | int 1–5 | Self-reported reliance level (1=minimal, 5=fully AI-generated) |
| `FinalOutcome` | categorical | Assignment Completed / Idea Drafted / Confused / Gave Up |
| `UsedAgain` | boolean | Would participant use AI for similar tasks again? |
| `SatisfactionRating` | float 1–5 | Post-session satisfaction with AI assistance |

**Key statistics:**
- Mean session length: 19.8 min (SD = 13.9)
- Mean prompts per session: 5.6 (SD = 4.6)
- Mean AI assistance level: 3.5 / 5 (SD = 0.99)
- Mean satisfaction: 3.4 / 5 (SD = 1.1)
- Overall task success rate: 76.3%
- Would use AI again: 70.6%

---

## Methodology Summary

- **Design:** Naturalistic observational panel study (N = 10,000 sessions)
- **Participants:** High School (20.3%), Undergraduate (59.8%), Graduate (19.9%)
- **Disciplines:** Biology, CS, Engineering, Math, Psychology, History, Business (~equal)
- **AI Reliance:** Self-reported 1–5 scale per session; corroborated by prompt count
- **Statistics:** Welch's t-tests, chi-square, one-way ANOVA, Pearson correlation, OLS and logistic regression, 5-fold cross-validated AUC
- **Effect sizes:** Cohen's d, Cramér's V, R²
- **Significance threshold:** α = 0.05 (Bonferroni-corrected for multiple comparisons)

Full methodology: [`methodology.md`](methodology.md) | Survey instrument: [`survey_questions.md`](survey_questions.md)

---

## Key Findings

### 1. AI Level Does Not Significantly Alter Session Duration
Sessions with high AI reliance (levels 4–5) averaged **19.7 min** vs **19.9 min** for low-reliance sessions. One-way ANOVA across all 5 levels was **not significant** (F = 0.67, p = ns, Cohen's d = 0.015 — *negligible* effect). AI assistance does not appear to compress task time; time efficiency may depend more on task type and user skill.

### 2. Overall Task Success Rate is 76.3%; AI Level Shows Weak Association
Three-quarters of sessions resulted in success (Assignment Completed or Idea Drafted). The chi-square test of AI assistance group × outcome was **not significant** (χ²(6) = 7.03, p = ns, Cramér's V = 0.019). This suggests that self-selected AI reliance level is not, by itself, a reliable predictor of task success — motivation and task difficulty are likely stronger confounders.

### 3. Overreliance Rate Slightly Elevated, But Not Strongly Group-Differentiated
High-AI sessions (levels 4–5) had a struggle rate of **23.4%** vs **23.3%** for low-AI sessions — a near-identical rate. The chi-square test was **not significant** (χ²(1) = 0.46, p = ns). This *null finding* is itself informative: students who used AI heavily did not fail *more* often than those who used it minimally, suggesting AI use may be appropriately calibrated to task difficulty.

### 4. Satisfaction is Strongly Correlated with AI Assistance Level (r = 0.776)
This is the study's strongest and most striking finding. Satisfaction ratings showed a large positive correlation with AI assistance level (r = 0.776, p < .001). OLS regression explained **60.2% of variance** in satisfaction. Participants who used AI more heavily *felt* their sessions were significantly more satisfying — even when outcome success rates were similar. This **satisfaction inflation** is a key signal of potential overreliance risk.

### 5. Task Success Strongly Predicts Willingness to Use AI Again (r = 0.371, p < .001)
70.6% of participants said they would use AI again for similar tasks. Successful task completion was the strongest predictor (r = 0.371). Satisfaction was not a significant predictor of repeat usage (r = -0.009, ns), suggesting behavioral intention is driven more by objective success than subjective enjoyment.

### 6. Task Type Effects on Satisfaction: No Significant Variation
One-way ANOVA on satisfaction across task types was **not significant** (F = 1.05, p = ns), indicating AI assistance is perceived as roughly equally valuable regardless of whether the task is Writing, Coding, Studying, etc.

### 7. Beginner vs. Expert Differences: Negligible
High School vs Graduate success rates differed by only **0.9 percentage points** (76.2% vs 75.3%, Cohen's d = 0.021 — negligible). Expert status does not confer measurable advantage (or disadvantage) in AI-assisted task outcomes, at least not at the level captured by this dataset.

### 8. Predictive Model Performance: Modest
A logistic regression model predicting session struggle from AI level, prompts, session length, student level, and task type achieved a 5-fold cross-validated **AUC = 0.534** — marginally above chance. Session-level features as captured here are insufficient to reliably predict individual outcomes, highlighting the importance of unobserved factors (task difficulty, content, prior knowledge).

---

## Statistical Summary Table

| Research Question | Test | Result | Effect Size | Significant? |
|---|---|---|---|---|
| RQ1: Session efficiency | ANOVA (5 levels) | F = 0.67 | Cohen's d = 0.015 | No |
| RQ2: Task success | Chi-square | χ²(6) = 7.03 | Cramér's V = 0.019 | No |
| RQ3: Overreliance | Chi-square | χ²(1) = 0.46 | — | No |
| RQ4: Task satisfaction | ANOVA | F = 1.05 | — | No |
| RQ5: Beginner vs expert | Welch's t | t-test | Cohen's d = 0.021 | No |
| RQ6: Satisfaction ↔ AI level | Pearson r | r = 0.776 | R² = 0.60 | Yes *** |
| RQ7: Struggle prediction | Logistic CV AUC | 0.534 | — | Marginally above chance |
| RQ8: Repeat usage ↔ success | Point-biserial | r = 0.371 | — | Yes *** |

---

## Charts

| Chart | Description |
|---|---|
| [`productivity_comparison.png`](charts/productivity_comparison.png) | Session length & prompt volume by AI assistance level |
| [`quality_scores.png`](charts/quality_scores.png) | Satisfaction ratings by task type and student level |
| [`trust_vs_accuracy.png`](charts/trust_vs_accuracy.png) | Trust (UsedAgain) and struggle rates by AI level |
| [`skill_retention.png`](charts/skill_retention.png) | Confusion/gave-up rates as skill retention risk indicators |
| [`task_category_effects.png`](charts/task_category_effects.png) | Success rates and satisfaction heatmap by task × AI level |
| [`confidence_gap.png`](charts/confidence_gap.png) | Satisfaction by outcome and AI level regression |

---

## Limitations

1. **Observational design** — no random assignment to AI conditions; causal claims are limited
2. **Self-reported measures** — AI reliance, outcome, and satisfaction are subjective
3. **No objective quality scoring** — output quality not rated by independent evaluators
4. **Tool heterogeneity** — multiple AI tools used; tool-level effects not isolatable
5. **No longitudinal retention testing** — skill retention approximated from session outcomes
6. **Convenience sample** — self-selected student volunteers; limited generalizability

---

## Future Work

- **Longitudinal skill assessment:** Pre/post testing of unassisted task performance to directly measure skill retention
- **Blind quality evaluation:** Independent expert scoring of session outputs to validate self-reported outcomes
- **Tool-level analysis:** Stratifying by specific AI tool (ChatGPT vs Claude vs Gemini) to detect tool effects
- **Intervention study:** Randomized assignment to AI reliance conditions to enable causal inference
- **Cognitive load measures:** EEG or dual-task paradigms to measure mental effort during AI-assisted tasks
- **Temporal trends:** Longitudinal analysis of how AI reliance patterns evolve over months of use

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI-Research.git
cd AI-Research

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis notebook
jupyter notebook analysis.ipynb

# 5. Or run the Python pipeline directly
python -c "
from src.data_cleaning import load_clean
from src.statistical_analysis import run_all
from src.visualization import generate_all_charts
df = load_clean()
results = run_all(df)
generate_all_charts(df)
"
```

All outputs (charts, analysis_results.json) will be regenerated automatically.

---

## Repository Structure

```
ai-productivity-study/
├── README.md                    # This file
├── dataset.csv                  # Raw dataset (10,000 sessions)
├── analysis.ipynb               # Full analysis notebook
├── requirements.txt             # Python dependencies
├── survey_questions.md          # Participant survey instrument
├── methodology.md               # Full research methodology
├── report.pdf                   # Publication-style research report
├── charts/
│   ├── productivity_comparison.png
│   ├── quality_scores.png
│   ├── trust_vs_accuracy.png
│   ├── skill_retention.png
│   ├── task_category_effects.png
│   └── confidence_gap.png
└── src/
    ├── data_cleaning.py         # Data loading, cleaning, feature engineering
    ├── statistical_analysis.py  # All inferential statistics
    ├── visualization.py         # Chart generation
    └── utils.py                 # Shared utilities
```

---

---

## License

MIT License. Dataset and code are open for research and educational use.

---

*Built with Python 3.11 · pandas · scipy · statsmodels · scikit-learn · matplotlib · seaborn*
