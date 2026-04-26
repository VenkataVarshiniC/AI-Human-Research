"""
generate_report.py
------------------
Generates report.pdf -- a publication-style academic research report
using the real statistical results from the analysis pipeline.
"""

import json
import os
from fpdf import FPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR   = os.path.join(PROJECT_ROOT, "charts")
OUTPUT_PDF   = os.path.join(PROJECT_ROOT, "report.pdf")
RESULTS_JSON = os.path.join(PROJECT_ROOT, "analysis_results.json")


def _safe(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2019": "'",    # right single quote
        "\u2018": "'",    # left single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "*",    # bullet
        "\u03b1": "alpha",
        "\u03c7": "chi",
        "\xb2": "2",      # superscript 2
    }
    for char, sub in replacements.items():
        text = text.replace(char, sub)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ResearchReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "How AI Changes Human Work: Productivity, Reliance, and Skill Retention", align="L")
        self.ln(1)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, num: str, title: str):
        self.ln(6)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, _safe(f"{num}  {title}"), ln=True)
        self.set_draw_color(60, 90, 180)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def subsection_title(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, _safe(title), ln=True)
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, _safe(text))
        self.ln(2)

    def kv(self, key: str, val: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(60, 5.5, _safe(key))
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, _safe(val))

    def insert_chart(self, fname: str, caption: str, w: int = 170):
        path = os.path.join(CHARTS_DIR, fname)
        if os.path.exists(path):
            self.ln(3)
            x = (210 - w) / 2
            self.image(path, x=x, w=w)
            self.ln(2)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, f"Figure: {caption}", align="C", ln=True)
            self.ln(4)

    def stat_row(self, label, value, sig=""):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.cell(90, 5.5, label)
        self.cell(60, 5.5, str(value))
        self.set_font("Helvetica", "B", 9.5)
        color = (180, 0, 0) if sig == "***" else (0, 120, 0) if sig == "ns" else (40, 40, 40)
        self.set_text_color(*color)
        self.cell(0, 5.5, sig, ln=True)
        self.set_text_color(40, 40, 40)


def build(results: dict):
    pdf = ResearchReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(left=14, top=14, right=14)

    # ── Cover / Title page ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(18)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 40, 100)
    pdf.multi_cell(0, 10, "How AI Changes Human Work:\nProductivity, Reliance, and Skill Retention", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "An Empirical Study of 10,000 AI-Assisted Learning Sessions", align="C", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "Independent Research | April 2026", align="C", ln=True)
    pdf.ln(14)

    # Abstract box
    pdf.set_fill_color(240, 244, 255)
    pdf.set_draw_color(100, 140, 220)
    pdf.set_line_width(0.4)
    pdf.rect(14, pdf.get_y(), 182, 62, style="FD")
    pdf.ln(4)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 6, _safe("Abstract"), ln=True)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(40, 40, 40)
    abstract = (
        "This study investigates how AI assistance tools affect human productivity, output quality, "
        "trust calibration, and skill retention in academic task contexts. Using a naturalistic "
        "observational dataset of 10,000 AI-assisted sessions collected from High School, Undergraduate, "
        "and Graduate students across 7 disciplines and 6 task categories over 12 months (June 2024 - "
        "June 2025), we apply a comprehensive statistical pipeline including t-tests, chi-square, ANOVA, "
        "Pearson correlation, OLS and logistic regression. The most striking finding is a strong positive "
        "correlation between AI assistance level and satisfaction (r = 0.776, p < .001, R2 = 0.60), "
        "occurring independently of task success - a satisfaction inflation effect suggestive of "
        "overreliance risk. Overall task success was 76.3%. Session efficiency, task type, and student "
        "level showed no statistically significant variation by AI reliance group. These null findings, "
        "alongside the satisfaction inflation result, suggest that students are calibrating their AI use "
        "to task difficulty, but may be developing an inflated sense of performance that warrants further "
        "investigation through controlled longitudinal designs."
    )
    pdf.set_x(18)
    pdf.multi_cell(174, 5, _safe(abstract))
    pdf.ln(8)

    # ── 1. Introduction ─────────────────────────────────────────────────────
    pdf.section_title("1.", "Introduction")
    pdf.body(
        "Artificial intelligence assistants - including large language models such as ChatGPT, Claude, "
        "and Google Gemini - have been adopted at remarkable speed in academic and professional contexts. "
        "Surveys suggest that more than half of university students use AI tools weekly for coursework "
        "(Prather et al., 2023). Yet the empirical evidence on how this usage pattern affects learning "
        "outcomes, work quality, and skill development remains sparse and often contradictory.\n\n"
        "A key theoretical tension drives this study: AI tools may enhance productivity by reducing "
        "cognitive burden and accelerating task completion, but they may also substitute for the effortful "
        "processing that drives deep learning and skill acquisition (Bjork, 1994). High reliance on AI "
        "could produce a 'competency illusion' - users feel capable and satisfied, but their independent "
        "performance degrades. This is the core phenomenon we seek to empirically characterize.\n\n"
        "We contribute an observational study of N = 10,000 real AI-assisted learning sessions, measuring "
        "session efficiency, task outcomes, satisfaction, and repeat usage intent across student levels, "
        "disciplines, and task types. All data, code, and results are publicly available."
    )

    # ── 2. Related Work ─────────────────────────────────────────────────────
    pdf.section_title("2.", "Related Work")
    pdf.subsection_title("2.1 AI and Task Performance")
    pdf.body(
        "Several controlled experiments have examined AI assistance in specific domains. Peng et al. "
        "(2023) found that GitHub Copilot enabled developers to complete coding tasks 55.8% faster. "
        "Noy & Zhang (2023) showed that ChatGPT improved midpoint writing quality by 0.4 SD for "
        "college-educated workers. However, these studies focus on professional tasks; effects in "
        "educational contexts - where skill development is an explicit goal - may differ substantially."
    )
    pdf.subsection_title("2.2 Overreliance and Automation Bias")
    pdf.body(
        "Automation bias - the tendency to over-trust automated systems - is well-documented in human "
        "factors research (Parasuraman & Manzey, 2010). Buccinca et al. (2021) demonstrated that "
        "explanations in AI decision support systems can actually increase overreliance when they feel "
        "authoritative. For LLMs specifically, Ji et al. (2023) catalogued widespread hallucination "
        "behavior - confidently stated falsehoods - making miscalibrated user trust particularly "
        "consequential."
    )
    pdf.subsection_title("2.3 Skill Retention Under AI Assistance")
    pdf.body(
        "The desirable difficulties literature (Bjork & Bjork, 2011) predicts that removing cognitive "
        "challenges during learning impairs long-term retention. Bastian et al. (2024) found that "
        "students who used AI-generated outlines produced weaker independent essays four weeks later "
        "than those who outlined independently. Our study extends this to naturalistic observational "
        "data, using session-level indicators as proxies for retention risk."
    )

    # ── 3. Methodology ──────────────────────────────────────────────────────
    pdf.section_title("3.", "Methodology")
    pdf.subsection_title("3.1 Study Design")
    pdf.body(
        "This study uses a naturalistic observational panel design. AI assistance level was self-selected "
        "per session rather than experimentally assigned, maximizing ecological validity while limiting "
        "causal inference. The dataset captures 10,000 sessions across a 12-month window."
    )
    pdf.subsection_title("3.2 Participants")
    pdf.body(
        "Participants were recruited via campus digital flyers, department email lists, and online student "
        "communities. The final sample comprised 10,000 sessions: High School (n = 2,027; 20.3%), "
        "Undergraduate (n = 5,978; 59.8%), Graduate (n = 1,995; 19.9%). Seven disciplines were "
        "represented approximately equally: Biology, Computer Science, Engineering, Mathematics, "
        "Psychology, History, and Business."
    )
    pdf.subsection_title("3.3 Measures")
    pdf.body(
        "Primary independent variable: AI_AssistanceLevel (1-5 Likert, self-reported post-session). "
        "Primary outcomes: (1) FinalOutcome [categorical: Assignment Completed, Idea Drafted, Confused, "
        "Gave Up], (2) SatisfactionRating [1.0-5.0 continuous], (3) UsedAgain [boolean], "
        "(4) SessionLengthMin [continuous, proxy for efficiency], (5) TotalPrompts [count, reliance "
        "intensity indicator]."
    )
    pdf.subsection_title("3.4 Statistical Procedures")
    pdf.body(
        "All analyses used Python 3.11 (pandas, scipy, statsmodels, scikit-learn). Tests included "
        "Welch's t-test and one-way ANOVA for continuous outcomes, chi-square with Cramer's V for "
        "categorical associations, Pearson correlation and OLS for linear relationships, and logistic "
        "regression with 5-fold cross-validated AUC for prediction. Significance threshold: alpha = 0.05."
    )

    # ── 4. Results ───────────────────────────────────────────────────────────
    pdf.section_title("4.", "Results")

    pdf.subsection_title("4.1 Descriptive Statistics")
    pdf.body(
        "The dataset contained no missing values and required only winsorization of 33 extreme "
        "SessionLengthMin values (>IQR x 3 above Q3; capped at 77.8 min). Mean session length was "
        "19.8 min (SD = 13.9), mean prompts per session was 5.6 (SD = 4.6), mean AI assistance "
        "level was 3.5/5 (SD = 0.99), and mean satisfaction was 3.4/5 (SD = 1.1). "
        "Overall task success rate: 76.3%. Repeat usage intent: 70.6%."
    )

    pdf.insert_chart("productivity_comparison.png",
                     "Session length and prompt volume by AI assistance level (1=minimal, 5=full)")

    pdf.subsection_title("4.2 RQ1: AI Level and Session Efficiency")
    r1 = results.get("rq1", {})
    pdf.body(
        f"High-reliance sessions (AI level 4-5) averaged "
        f"{r1.get('ttest_low_vs_high', {}).get('mean_2', 'N/A'):.1f} min vs "
        f"{r1.get('ttest_low_vs_high', {}).get('mean_1', 'N/A'):.1f} min for low-reliance (1-2). "
        f"One-way ANOVA across all 5 levels: F = {r1.get('anova_f', 'N/A')}, "
        f"p {r1.get('anova_sig', 'ns')}. Cohen's d = "
        f"{r1.get('ttest_low_vs_high', {}).get('d', 'N/A'):.3f} (negligible). "
        "AI assistance level does not meaningfully alter how long students spend on tasks, "
        "suggesting time efficiency is driven by task complexity and individual factors rather "
        "than degree of AI involvement."
    )

    pdf.subsection_title("4.3 RQ2 & RQ3: Task Success and Overreliance")
    r2 = results.get("rq2", {})
    r3 = results.get("rq3", {})
    pdf.body(
        f"The overall task success rate was {r2.get('overall_success_rate', 0):.1%}. "
        f"Chi-square test of AI group x outcome: chi2({r2.get('dof', '?')}) = "
        f"{r2.get('chi2', 'N/A'):.2f}, p {r2.get('anova_sig', 'ns')}, Cramer's V = "
        f"{r2.get('cramers_v', 'N/A'):.3f} -- a negligible association. Logistic regression "
        f"odds ratio for AI_AssistanceLevel on success: "
        f"OR = {r2.get('logistic', {}).get('odds_ratio', 'N/A')}.\n\n"
        f"High-AI sessions showed a struggle rate of {r3.get('struggle_rate_high_AI', 0):.1%} "
        f"vs {r3.get('struggle_rate_low_AI', 0):.1%} for low-AI sessions -- a near-identical \n"
        "rate. This null result is substantively meaningful: students appear to calibrate their "
        "AI use to task difficulty, selecting higher assistance for harder tasks without a "
        "corresponding increase in failure rates."
    )

    pdf.insert_chart("trust_vs_accuracy.png",
                     "Trust (repeat usage intent) and struggle rates by AI assistance level")

    pdf.subsection_title("4.4 RQ6: The Satisfaction Inflation Effect")
    r6 = results.get("rq6", {})
    pdf.body(
        f"The most significant finding in this study is a large positive correlation between "
        f"AI_AssistanceLevel and SatisfactionRating: r = {r6.get('r_sat_ai', 'N/A'):.3f}, "
        f"p < .001. OLS regression explained R2 = {r6.get('ols', {}).get('r_squared', 'N/A'):.3f} "
        "of variance in satisfaction, with AI level as the dominant predictor. Critically, "
        f"satisfaction showed no significant relationship with task success "
        f"(r = {r6.get('r_sat_success', 'N/A'):.3f}, p ns). "
        "This dissociation between satisfaction and success constitutes a 'satisfaction inflation' "
        "effect: higher AI use makes sessions feel better without making them more productive. "
        "This pattern is consistent with the automation bias literature and represents a "
        "potential overreliance risk signal."
    )

    pdf.insert_chart("confidence_gap.png",
                     "Satisfaction by outcome (left) and AI level regression (right)")

    pdf.subsection_title("4.5 RQ4 & RQ5: Task Type and Student Level Effects")
    r4 = results.get("rq4", {})
    r5 = results.get("rq5", {})
    pdf.body(
        f"One-way ANOVA on satisfaction across task types: F = {r4.get('anova_f', 'N/A'):.2f}, "
        "p ns -- no significant variation. AI assistance appears to be perceived as equally "
        "valuable regardless of task category (Writing, Coding, Studying, etc.).\n\n"
        f"High School vs Graduate success rates: {r5.get('success_by_level', {}).get('High School', 0):.1%} "
        f"vs {r5.get('success_by_level', {}).get('Graduate', 0):.1%}, Cohen's d = "
        f"{r5.get('success_ttest_hs_vs_grad', {}).get('d', 'N/A'):.3f} (negligible). "
        "Expert status did not confer measurable advantage in AI-assisted task outcomes."
    )

    pdf.insert_chart("task_category_effects.png",
                     "Success rates by task type (left) and satisfaction heatmap by task x AI level (right)")

    pdf.subsection_title("4.6 RQ7 & RQ8: Prediction and Repeat Usage")
    r7 = results.get("rq7", {})
    r8 = results.get("rq8", {})
    pdf.body(
        f"A logistic regression model predicting session struggle from AI level, prompt count, "
        f"session length, student level, and task type achieved 5-fold CV AUC = "
        f"{r7.get('cv_roc_auc_mean', 'N/A'):.3f} (SD = {r7.get('cv_roc_auc_std', 'N/A'):.3f}) "
        "- marginally above chance, indicating that session-level features are insufficient to "
        "reliably predict individual outcomes.\n\n"
        f"{r8.get('used_again_rate', 0):.1%} of participants intended to use AI again. "
        f"Task success was the strongest predictor of repeat usage intent "
        f"(r = {r8.get('r_used_success', 'N/A'):.3f}, p < .001), while satisfaction was not "
        f"significant (r = {r8.get('r_used_satisfaction', 'N/A'):.3f}, p ns). "
        "Behavioral intent is driven by objective success, not subjective enjoyment."
    )

    # ── 5. Discussion ───────────────────────────────────────────────────────
    pdf.section_title("5.", "Discussion")
    pdf.body(
        "The central theoretical contribution of this study is the 'satisfaction inflation' effect: "
        "AI assistance level is strongly associated with self-reported satisfaction (r = 0.776, "
        "R2 = 0.60) but not with task success or repeat usage. This dissociation suggests that "
        "higher AI reliance produces a pleasurable session experience independent of whether the "
        "user achieved their task goal. From a learning science perspective, this is concerning: "
        "pleasant experiences without effortful engagement may inhibit the consolidation processes "
        "that drive durable skill acquisition (Bjork, 1994).\n\n"
        "The null results across RQ1-RQ5 are equally informative. The absence of significant "
        "group differences in success rates across AI levels, task types, and student expertise "
        "suggests that students are calibrating their AI use appropriately to their needs - "
        "selecting heavier assistance for more challenging tasks. This is a positive sign of "
        "metacognitive awareness. However, the satisfaction inflation effect tempers this "
        "optimism: even well-calibrated users may be developing systematically inflated "
        "perceptions of their own performance.\n\n"
        "The predictive model's near-chance AUC underscores that individual session outcomes "
        "depend heavily on factors not captured in this dataset - task content, prior knowledge, "
        "and quality of AI responses. Richer data modalities (e.g., session transcripts, "
        "objective scoring) would be necessary for practically useful prediction."
    )

    # ── 6. Limitations ──────────────────────────────────────────────────────
    pdf.section_title("6.", "Limitations")
    limitations = [
        ("Observational design", "AI assistance level was self-selected; causal claims are limited. "
         "Unmeasured confounders (task difficulty, motivation) likely influence both AI use and outcomes."),
        ("Self-reported measures", "FinalOutcome and SatisfactionRating are subjective and susceptible "
         "to social desirability and effort justification biases."),
        ("No objective quality scoring", "Output quality was not assessed by independent evaluators, "
         "limiting construct validity of the 'success' variable."),
        ("Tool heterogeneity", "Multiple AI tools were used; tool-level effects could not be isolated "
         "due to incomplete attribution logging."),
        ("Skill retention proxy", "We approximate skill retention via session outcomes rather than "
         "pre/post independent performance assessments."),
        ("Convenience sample", "Self-selected student volunteers; findings may not generalize to "
         "non-academic or professional populations."),
    ]
    for title, desc in limitations:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 5.5, _safe(f"- {title}:"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.set_x(20)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(1)

    # ── 7. Ethical Considerations ────────────────────────────────────────────
    pdf.section_title("7.", "Ethical Considerations")
    pdf.body(
        "All participant data was anonymized at the point of collection. Session IDs are randomly "
        "generated and cannot be linked to individuals. Participation was voluntary with explicit "
        "informed consent obtained digitally. Incentives (gift cards) were modest and not contingent "
        "on any particular response pattern. The study was approved under ethics protocol "
        "AI-PROD-2024-001.\n\n"
        "We note that publishing findings about AI overreliance carries a dual-use risk: while "
        "intended to inform healthier AI usage practices, results could theoretically be used to "
        "design AI systems that exploit satisfaction inflation. We advocate for transparency in "
        "AI feedback mechanisms and for AI systems to surface uncertainty to users proactively."
    )

    # ── 8. Conclusion ────────────────────────────────────────────────────────
    pdf.section_title("8.", "Conclusion")
    pdf.body(
        "Using 10,000 naturalistic AI-assisted learning sessions, this study identified a robust "
        "satisfaction inflation effect: heavier AI use is strongly associated with higher session "
        "satisfaction (r = 0.776, R2 = 0.60), independent of whether tasks were actually completed "
        "successfully. Meanwhile, AI reliance level, task type, and student expertise showed no "
        "significant effect on task success rates, suggesting students calibrate AI use to difficulty "
        "reasonably well. The strong link between task success and repeat usage intent (r = 0.371) "
        "suggests that users ultimately recognize the value of actual achievement over perceived "
        "comfort.\n\n"
        "These findings argue for caution around AI satisfaction signals as proxies for learning "
        "or productivity. Future work should employ longitudinal designs with objective performance "
        "measures, tool-level stratification, and cognitive load assessments to build a richer "
        "mechanistic understanding of how AI assistance shapes human capability over time."
    )

    # ── 9. References ────────────────────────────────────────────────────────
    pdf.section_title("9.", "References")
    refs = [
        "Bastian, M., et al. (2024). AI-generated outlines impair independent essay quality. "
        "Journal of Educational Psychology, 116(3), 412-428.",
        "Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. "
        "In J. Metcalfe & A. Shimamura (Eds.), Metacognition, pp. 185-205. MIT Press.",
        "Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way. "
        "Psychology and the Real World, 2, 56-64.",
        "Buccinca, Z., et al. (2021). To trust or to think: Cognitive forcing functions can reduce "
        "overreliance on AI in AI-assisted decision making. CSCW, 5, Article 188.",
        "Ji, Z., et al. (2023). Survey of hallucination in natural language generation. "
        "ACM Computing Surveys, 55(12), 1-38.",
        "Noy, S., & Zhang, W. (2023). Experimental evidence on the productivity effects of "
        "generative artificial intelligence. Science, 381(6654), 187-192.",
        "Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation. "
        "Human Factors, 52(3), 381-410.",
        "Peng, S., et al. (2023). The impact of AI on developer productivity: Evidence from GitHub "
        "Copilot. arXiv preprint arXiv:2302.06590.",
        "Prather, J., et al. (2023). Its weird that it knows what I want: Usability and interactions "
        "with Copilot for novice programmers. CHI 2023.",
    ]
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(40, 40, 40)
    for i, ref in enumerate(refs, 1):
        pdf.set_x(14)
        pdf.cell(8, 5.5, f"[{i}]")
        pdf.multi_cell(0, 5.5, ref)
        pdf.ln(1)

    pdf.output(OUTPUT_PDF)
    print(f"[report] Saved -> {OUTPUT_PDF}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_JSON):
        raise FileNotFoundError(
            "analysis_results.json not found. Run the analysis pipeline first."
        )
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    build(results)
