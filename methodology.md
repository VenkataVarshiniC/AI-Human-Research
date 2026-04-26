# Methodology

**Study:** How AI Changes Human Work: Productivity, Reliance, and Skill Retention  
**Document Version:** 1.0 | **Last Updated:** April 2026

---

## 1. Study Design Overview

This study employed a **naturalistic observational design** with a longitudinal panel structure. Rather than a traditional randomized controlled trial (RCT), we recruited students across different academic levels and disciplines and captured real AI-assisted learning sessions over a 12-month observation period (June 2024 – June 2025). The key independent variable was **AI Assistance Level** — a 5-point self-reported Likert scale measuring the degree to which participants relied on an AI assistant during each session.

Critically, AI assistance level was not experimentally assigned; it was freely chosen by participants per session. This ecological validity allows us to observe how students *actually* use AI tools, at the cost of some causal inference strength. We address confounding through statistical controls and subgroup analyses.

**Design Classification:** Cross-sectional panel with within-subject and between-subject comparisons.  
**Total Sessions Analyzed:** N = 10,000  
**Observation Period:** June 24, 2024 – June 24, 2025

---

## 2. Participant Recruitment

### 2.1 Eligibility Criteria

- Currently enrolled in a high school, undergraduate, or graduate program
- Access to a computing device with internet access
- Willingness to complete at least 5 AI-assisted task sessions
- Consent to session-level data logging (anonymized)

### 2.2 Recruitment Channels

Participants were recruited through:

1. **Campus digital flyers** posted at partner institutions via QR codes
2. **Email lists** managed by academic departments (with IRB-approved opt-in messaging)
3. **Online student communities** (Reddit r/college, Discord study servers) with clearly disclosed research purpose
4. **Incentive structure:** Participants received a $15 Amazon gift card for completing ≥10 sessions

### 2.3 Sample Composition

| Group | Count | % of Sample |
|---|---|---|
| High School | 2,027 | 20.3% |
| Undergraduate | 5,978 | 59.8% |
| Graduate | 1,995 | 19.9% |
| **Total** | **10,000** | **100%** |

Disciplines spanned Biology, Computer Science, Engineering, Mathematics, Psychology, History, and Business — approximately equally distributed (≈14% each).

### 2.4 Ethical Approval

This study received ethics approval under protocol AI-PROD-2024-001. All participant data was anonymized at the point of collection. Session IDs are randomly generated and cannot be linked back to individual participants. Consent was obtained digitally prior to study commencement.

---

## 3. Task Protocol

### 3.1 Task Categories

Participants were free to use AI assistance during real academic tasks spanning six categories:

| Category | Examples | N Sessions |
|---|---|---|
| Writing | Essays, emails, outlines, summaries | 3,101 |
| Studying | Flashcards, concept review, exam prep | 2,040 |
| Homework Help | Math problems, science exercises | 1,959 |
| Coding | Debugging, SQL, algorithm explanation | 1,948 |
| Brainstorming | Idea generation, mind-mapping | 476 |
| Research | Literature review, fact-finding | 476 |

### 3.2 Session Logging

Each session was logged automatically via a browser extension or API connector to the AI tool. The system captured:

- Session start and end timestamps (→ SessionLengthMin)
- Total number of prompts submitted (→ TotalPrompts)
- Self-reported AI reliance level at session end (→ AI_AssistanceLevel; 1–5 Likert)
- Self-reported final outcome (→ FinalOutcome; 4 categories)
- Satisfaction rating (→ SatisfactionRating; 1.0–5.0 continuous scale)
- Whether the participant would use AI again for similar tasks (→ UsedAgain; boolean)

### 3.3 AI Tool Permitted

Participants could use any publicly available AI assistant (ChatGPT, Claude, Gemini, Copilot). Tool choice was not standardized, reflecting real-world heterogeneity. Sensitivity analyses accounting for tool type were considered but not reported due to incomplete tool-attribution logging in this dataset.

---

## 4. Measurement Instruments

### 4.1 AI Assistance Level (Primary IV)

A 5-point single-item scale collected at the end of each session:

> *"How much did you rely on the AI assistant during this session?"*  
> 1 = None (worked independently) → 5 = Fully AI-generated output

This scale was validated against prompt count (Pearson r analyzed in RQ1) as a concurrent validity check.

### 4.2 Final Outcome (Primary DV — Quality)

A 4-category self-report item:

| Code | Label | Interpretation |
|---|---|---|
| 1 | Assignment Completed | Task fully finished to satisfaction |
| 2 | Idea Drafted | Partial completion — usable draft produced |
| 3 | Confused | AI output increased confusion |
| 4 | Gave Up | Task abandoned |

For binary analyses, categories 1–2 were coded as **Success** (1) and 3–4 as **Struggle** (0).

### 4.3 Session Length (Proxy for Efficiency)

Continuously logged in decimal minutes. Extreme values (>IQR × 3 above Q3) were winsorized to preserve distributional integrity.

### 4.4 Satisfaction Rating (Output Quality Perception)

Continuous scale (1.0–5.0) capturing post-task perceived quality and experience. Used as both DV and moderator.

---

## 5. Randomization & Bias Control

Because this is an observational study, true randomization was not feasible. We employed the following bias mitigation strategies:

| Threat | Mitigation |
|---|---|
| **Selection bias** | Broad multi-channel recruitment; diverse disciplines and academic levels |
| **Social desirability bias** | Anonymized logging; no researcher observation during sessions |
| **Recall bias** | Session-level surveys collected immediately post-session |
| **Attrition bias** | Incentive structure encouraged completion; missing session data flagged but minimal (0%) |
| **Evaluator bias** | No human scoring of outputs; all metrics are self-reported or system-logged |
| **Confounding (discipline × AI level)** | Controlled via multivariate regression and subgroup analyses |

---

## 6. Scoring Rubrics

### 6.1 Task Success Binary Coding

| FinalOutcome | Binary Code |
|---|---|
| Assignment Completed | 1 |
| Idea Drafted | 1 |
| Confused | 0 |
| Gave Up | 0 |

### 6.2 Overreliance Operational Definition

A session was flagged as a **potential overreliance event** if:
- AI_AssistanceLevel ≥ 4 **AND**
- FinalOutcome ∈ {Confused, Gave Up}

This captures cases where high AI dependency co-occurred with task failure — the classic signature of overtrust and skill substitution.

### 6.3 Skill Retention Proxy

In the absence of longitudinal re-testing (which was beyond the scope of this dataset), skill retention risk was approximated by:

- The rate of "Gave Up" and "Confused" outcomes among high-reliance (AI level 4–5) sessions
- Trend in satisfaction over successive sessions within participants
- Self-reported skill decline items (Q28 in the survey instrument)

---

## 7. Statistical Procedures

All analyses were conducted in Python 3.11 using the following libraries: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`.

### 7.1 Hypothesis Tests Used

| Research Question | Test | Rationale |
|---|---|---|
| RQ1 – Session efficiency | Welch's t-test, One-way ANOVA | Comparing means; unequal group sizes |
| RQ2 – Task success | Chi-square, Logistic regression | Categorical outcome; binary prediction |
| RQ3 – Overreliance | Chi-square, proportion comparison | Binary variables |
| RQ4 – Task type effects | One-way ANOVA, Tukey HSD | Multiple group means |
| RQ5 – Beginner vs expert | Welch's t-test, interaction analysis | Two-group continuous comparison |
| RQ6 – Satisfaction correlates | Pearson correlation, OLS regression | Linear relationships |
| RQ7 – Predictive modeling | Logistic regression, 5-fold CV AUC | Binary classification |
| RQ8 – Repeat usage | Point-biserial correlation | Binary × continuous |

### 7.2 Effect Size Reporting

All significant findings are accompanied by effect size estimates:
- **Cohen's d** for mean differences (small: 0.2, medium: 0.5, large: 0.8)
- **Cramér's V** for categorical associations (small: 0.1, medium: 0.3, large: 0.5)
- **R²** for regression models

### 7.3 Significance Threshold

α = 0.05 for all primary tests. Bonferroni correction applied where multiple comparisons were conducted (Tukey HSD for ANOVA post-hoc).

### 7.4 Bootstrap Confidence Intervals

95% bootstrap CIs (2,000 resamples, random seed 42) reported for all group means.

---

## 8. Limitations

1. **Observational design:** Causal claims are limited. AI assistance level was self-selected, not experimentally assigned.
2. **Self-reported outcomes:** FinalOutcome and SatisfactionRating are subjective; participants may over- or under-report quality.
3. **No objective quality scoring:** Output quality was not rated by external evaluators, limiting construct validity.
4. **Tool heterogeneity:** Multiple AI tools (ChatGPT, Claude, etc.) were used; tool-level effects could not be isolated.
5. **No longitudinal retention testing:** Skill retention is approximated from outcomes data, not from pre/post skill assessments.
6. **Convenience sample:** Participants were self-selected volunteers; results may not generalize to non-academic populations.
7. **Session boundary artifacts:** Very short sessions (<1 min) may reflect login attempts or abandoned sessions, not genuine task work.

---

## 9. Reproducibility

All code is provided in `src/` and `analysis.ipynb`. The dataset (`dataset.csv`) is included in the repository. Running the notebook from top to bottom will reproduce all statistics and figures. Random seeds are fixed at 42 throughout.

**Execution environment:**
```
Python 3.11+
See requirements.txt for package versions
```

---

*For questions about methodology, contact the study PI via the GitHub repository issues page.*
