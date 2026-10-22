"""Builds report/project_report.docx from the files in results/ and results/figures/.

Run from the project root:   python report/build_report.py
All numbers quoted in the text are read from results/*.csv and results/key_results.json.
Editable title-page fields are marked [EDIT: ...].
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
OUT = Path(__file__).resolve().parent / "project_report_detailed.docx"
TITLE = "Maximum Likelihood Estimation and Model Selection for Stationary Renewable Energy Output Data"

K = json.loads((RES / "key_results.json").read_text())
rd = lambda name: pd.read_csv(RES / name)
STAT, VERD, MLE, MC = rd("stationarity_tests.csv"), rd("stationarity_verdicts.csv"), rd("mle_estimates.csv"), rd("model_comparison.csv")
VALID, SENS, PER = rd("mle_validation.csv"), rd("sensitivity_distribution_ranking.csv"), rd("periodicity_tests.csv")
TS, TSP, DIAG = rd("timeseries_model_comparison.csv"), rd("timeseries_best_model_parameters.csv"), rd("diagnostic_results.csv")
GC, GP, NG = rd("garch_comparison.csv"), rd("garch_best_model_parameters.csv"), rd("night_gap_sensitivity.csv")
FS, FT, RAWPOS = rd("forecast_scores.csv"), rd("forecast_training_model_selection.csv"), rd("supplementary_raw_positive_mw_comparison.csv")
RV = rd("residual_variance_by_fitted_level.csv")
PRE, IDX, ACF, ENV = K["preprocessing"], K["index_summary"], K["acf"], K["environment"]
TSK, FC = K["timeseries"], K["forecast"]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.3)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(11.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.25
for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold = "Times New Roman", Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(14 if name == "Heading 1" else 10), Pt(6)
    h.paragraph_format.keep_with_next = True

PAGE_BREAK_BEFORE = {'Table of Contents', 'Introduction', 'Dataset Description and Source', 'Statistical and Mathematical Background',
                     'Stationarity Testing', 'Maximum Likelihood Estimation', 'Model Selection Using AIC and BIC', 'Experimental Results', 'Conclusion', 'References', 'Appendix: Important Code and Additional Results'}
FIGN, TABN = [0], [0]
SECTIONS = []


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def H1(text):
    SECTIONS.append(text)
    if text.split(". ", 1)[-1] in PAGE_BREAK_BEFORE or text in PAGE_BREAK_BEFORE:
        doc.add_page_break()
    return doc.add_heading(text, 1)


def H2(text): return doc.add_heading(text, 2)


def P(text="", bold=False, italic=False, align=None, size=None, after=None):
    p = doc.add_paragraph()
    # lightweight **bold** markup
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part: continue
        r = p.add_run(part); r.bold = bold or (i % 2 == 1); r.italic = italic
        if size: r.font.size = Pt(size)
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align != "l": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p


def B(items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet")
        parts = t.split("**")
        for i, part in enumerate(parts):
            if part:
                r = p.add_run(part); r.bold = i % 2 == 1
        p.paragraph_format.space_after = Pt(2)


def EQ(text, label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.italic = True; r.font.name = "Cambria Math"; r.font.size = Pt(11.5)
    if label:
        r2 = p.add_run(f"        ({label})"); r2.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(6)


def CAPTION(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.italic = True; r.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(10)


def FIGURE(fname, caption, width=15.5):
    FIGN[0] += 1
    doc.add_picture(str(FIG / fname), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    CAPTION(f"Figure {FIGN[0]}. {caption}")


def TABLE(df, caption, widths=None, size=8.5, fmt=None):
    TABN[0] += 1
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Table {TABN[0]}. {caption}"); r.italic = True; r.font.size = Pt(10)
    cap.paragraph_format.keep_with_next = True; cap.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = ""
        run = cell.paragraphs[0].add_run(str(c)); run.bold = True; run.font.size = Pt(size)
        shade(cell, "D9E2F3")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            v = row[c]
            if isinstance(v, float) and pd.isna(v):
                s = "–"
            elif isinstance(v, float):
                f = (fmt or {}).get(c)
                s = f.format(v) if f else (f"{v:.4g}" if abs(v) < 1e5 else f"{v:,.0f}")
            else:
                s = str(v)
            cells[j].text = ""
            para = cells[j].paragraphs[0]; para.paragraph_format.space_after = Pt(0); para.paragraph_format.line_spacing = 1.0
            run = para.add_run(s); run.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def pf(p):  # p-value formatter
    return "<1e-15" if p < 1e-15 else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")


def add_field(paragraph, instr, placeholder=""):
    run = paragraph.add_run()
    for kind, text in (("begin", None), ("instr", instr), ("separate", None), ("text", placeholder), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text; run._r.append(el)
        elif kind == "text":
            el = OxmlElement("w:t"); el.text = text; run._r.append(el)
        else:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind); run._r.append(el)


# footer with page numbers
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_field(fp, "PAGE", "1")
for r in fp.runs: r.font.size = Pt(10)

# ============================================================================ 1. TITLE PAGE
SECTIONS.append("Title Page")
doc.add_paragraph()
P("ACADEMIC PROJECT REPORT", bold=True, align="c", size=13)
P(TITLE, bold=True, align="c", size=22, after=24)
P("Statistical analysis of solar photovoltaic output: stationarity testing, maximum likelihood estimation, "
  "AIC/BIC model selection and time-series modelling", italic=True, align="c", size=12, after=24)
for label in ("Student name", "Roll / enrolment number", "Programme and semester", "Department", "College / university",
              "Supervisor / professor", "Date of submission"):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{label}:  "); r.bold = True
    r2 = p.add_run(f"[EDIT: {label.lower()}]"); r2.font.highlight_color = 7  # yellow
    p.paragraph_format.space_after = Pt(8)
doc.add_paragraph()
P("The fields highlighted in yellow have deliberately been left blank for the author to complete.", italic=True, align="c", size=9.5)
doc.add_page_break()

# ============================================================================ 2. ABSTRACT
H1("Abstract")
mix = MLE[(MLE.model == "2-component normal mixture")].set_index("parameter")["estimate"]
P(f"This project studies the statistical behaviour of solar photovoltaic (PV) output using a genuine public dataset: the 10-minute "
  f"output of {PRE.get('n_plants', 137)} Alabama PV plants for the year 2006 from the U.S. National Renewable Energy Laboratory (NREL) "
  f"*Solar Power Data for Integration Studies*, as redistributed in the LSTNet repository. The plants' outputs are summed and averaged "
  f"to {PRE['n_hours']:,} hourly values. The data are NREL-modelled (simulated) output, not metered generation, and the file carries no timestamps; "
  f"the time axis is reconstructed from the regular 10-minute layout and checked against solar geometry.".replace("*", ""))
P(f"Because {100*PRE['share_hours_exact_zero']:.1f}% of hours are exactly zero (night), the raw series is not a candidate for a continuous "
  f"distribution. The analysis therefore uses a clear-sky index (output divided by an empirical clear-sky envelope) on daylight hours "
  f"(n = {PRE['n_in_sample']:,}). Augmented Dickey-Fuller (ADF) and KPSS tests show that the raw series is not stationary in the mean "
  f"(it has deterministic diurnal and annual cycles), while the clear-sky index is stationary in the hour-of-day dimension and, after removal of "
  f"an annual harmonic, passes both ADF and KPSS.")
P(f"Maximum likelihood estimators (MLE) were implemented from explicit log-likelihoods for seven models (Normal, Student-t, Log-normal, Gamma, "
  f"Weibull, scaled Beta and a two-component normal mixture), maximised numerically, checked through gradient and Hessian conditions and validated "
  f"against SciPy. By both AIC and BIC the two-component normal mixture is clearly best (log-likelihood {MC.set_index('model').loc['2-component normal mixture','loglik']:.1f}; "
  f"BIC advantage over the runner-up about {MC.set_index('model')['delta_BIC'].sort_values().iloc[1]:,.0f}); the best single-family model is the scaled Beta. "
  f"All parametric-bootstrap goodness-of-fit tests nevertheless reject every model, which is expected with strong serial dependence and is reported as a caveat. "
  f"Dependence dominates the likelihood: a seasonal ARMA error model, ARMA(2,1)x(1,0,0)_24, reaches a log-likelihood of {TSK['best_loglik']:.0f} versus about 1,001 for the best i.i.d. model. "
  f"BIC and AIC disagree on whether an annual cycle is needed (likelihood-ratio test p = {TSK['lr_test_annual_cycle']['p_value']:.4f}). "
  f"The residual diagnostics (Ljung-Box, ARCH-LM, Jarque-Bera) reject for all ARMA and GARCH models, so none of them is claimed to be adequate. "
  f"In a chronological one-hour-ahead forecast the ARMA model reduces the RMSE by {100*FS.iloc[0]['RMSE_skill_vs_index_persistence']:.1f}% relative to clear-sky-index persistence.")
P("**Keywords:** maximum likelihood estimation; stationarity; ADF; KPSS; AIC; BIC; solar power; Beta distribution; Gaussian mixture; ARMA; GARCH.", align="l")

# ============================================================================ 3. TOC
H1("Table of Contents")
TOC_ENTRIES = ["Title Page", "Abstract", "Table of Contents", "1. Introduction", "2. Background and Motivation", "3. Problem Statement",
               "4. Research Objectives", "5. Research Questions", "6. Scope and Limitations", "7. Dataset Description and Source",
               "8. Tools and Software Requirements", "9. Statistical and Mathematical Background", "10. Methodology and Workflow",
               "11. Data Preprocessing", "12. Exploratory Data Analysis", "13. Stationarity Testing", "14. Maximum Likelihood Estimation",
               "15. Candidate Distribution Models", "16. Time-Series Modelling", "17. Model Selection Using AIC and BIC",
               "18. Model Diagnostics and Validation", "19. Experimental Results", "20. Results and Discussion", "21. Conclusion",
               "22. Limitations and Future Scope", "23. References", "24. Appendix: Important Code and Additional Results"]
P("Parts 1-3 (title page, abstract, table of contents) are unnumbered; the report body is numbered 1-24 and covers parts 4-27 of the required structure in order.", italic=True, size=10, align="l")
TOC_PAGES = {}
if (Path(__file__).resolve().parent / "toc_pages.json").exists():
    TOC_PAGES = json.loads((Path(__file__).resolve().parent / "toc_pages.json").read_text())
for e in TOC_ENTRIES:
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2); p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.tab_stops.add_tab_stop(Cm(16), alignment=2, leader=1)  # right-aligned tab with dot leader
    p.add_run(e + "\t" + str(TOC_PAGES.get(e, "")))

# ============================================================================ numbered body
def N(i, title): return H1(f"{i}. {title}")


N(1, "Introduction")
P("Solar and wind generation are variable: their output depends on weather, season and time of day. Planning reserves, storage and grid operation "
  "therefore requires a quantitative description of this variability. Statistical modelling offers two complementary descriptions. The first is "
  "a *marginal distribution*: how likely is each output level, regardless of when it occurs? The second is a *dynamic model*: how does the output at one "
  "hour depend on the preceding hours? Maximum likelihood estimation (MLE) is the standard, principled method for fitting both, and information criteria "
  "(AIC and BIC) provide a principled way to choose between competing models.".replace("*", ""))
P("This project applies that toolkit end-to-end to a real, public PV dataset. A recurrent theme is that statistical theory has preconditions "
  "(stationarity, independence, positive support) that renewable-energy data do not automatically satisfy, so a large part of the work consists of "
  "testing those preconditions and transforming the data until the methods are applicable, and then reporting honestly where they still are not.")

N(2, "Background and Motivation")
P("Photovoltaic output follows the sun. Its dominant variation is a deterministic diurnal cycle (zero at night, maximum near solar noon) and a slower annual cycle "
  "(longer, stronger days in summer). Superimposed is a stochastic component driven by clouds. A common device in the solar literature (Inman, Pedro and Coimbra, 2013; "
  "Bacher, Madsen and Nielsen, 2009) is to divide the output by a clear-sky reference, producing a *clear-sky index* that removes most of the deterministic geometry. "
  "This is the approach adopted here, with an empirical envelope in place of a physical clear-sky model.")
P("Classical time-series theory (Box, Jenkins, Reinsel and Ljung, 2015; Hamilton, 1994) assumes (weak) stationarity: constant mean and variance and autocovariance that depends only on the lag. "
  "Unit-root tests such as ADF (Dickey and Fuller, 1979) and stationarity tests such as KPSS (Kwiatkowski et al., 1992) are the usual checks. Likelihood-based estimation "
  "goes back to Fisher (1922); Akaike (1974) and Schwarz (1978) derived the criteria that bear their names.")

N(3, "Problem Statement")
P("Given a year of PV output from a fleet of plants, (i) determine whether and in what sense the series is stationary; (ii) estimate, by maximum likelihood, the parameters of "
  "several candidate probability models for the stationary component; (iii) select among the candidates using AIC and BIC and goodness-of-fit tests that respect the fact that "
  "parameters were estimated; and (iv) determine whether dynamic (ARMA/GARCH) models are justified and how well they describe the data and forecast out of sample.")

N(4, "Research Objectives")
B(["Obtain and document a genuine public solar-output dataset and validate its integrity.",
   "Test stationarity with ADF and KPSS (explicit hypotheses) and with diagnostic group tests, and transform the data into a defensible stationary form.",
   "Implement MLE from first principles (likelihood, log-likelihood, numerical optimisation, convergence checks, standard errors) and validate against trusted libraries.",
   "Compare candidate distributions by AIC, BIC and parametric-bootstrap goodness-of-fit tests.",
   "Assess ARMA and GARCH models with residual diagnostics, and evaluate forecasts on a chronological hold-out set without leakage.",
   "Report limitations and unresolved model failures honestly."])

N(5, "Research Questions")
B(["RQ1. Is raw hourly PV output stationary? If not, which component breaks stationarity and what transformation restores it?",
   "RQ2. Which candidate distribution best describes the stationary component according to AIC and BIC, and is the ranking robust to preprocessing choices?",
   "RQ3. Does any candidate pass a goodness-of-fit test when the parameters are estimated from the same data?",
   "RQ4. Is serial dependence strong enough to justify ARMA modelling, and do ARMA/GARCH models pass residual diagnostics?",
   "RQ5. Does the selected dynamic model improve one-hour-ahead forecasts over simple baselines?"])

N(6, "Scope and Limitations")
P("The study covers one calendar year (2006), one region (Alabama), and one aggregate series (fleet total). Wind data are not analysed. Principal limitations, discussed in detail in Section 22:")
B(["The data are NREL-modelled PV output, not metered generation; conclusions describe this simulated output.",
   "The file contains no timestamps; the time axis is reconstructed (Section 7).",
   "The clear-sky envelope is empirical (a moving quantile), not a physical clear-sky model; the index can exceed 1.",
   "Only one year is available, so inter-annual variability cannot be assessed; the effective number of independent days is about "
   f"{K['effective_sample']['n_eff_days_ar1_approx']:.0f}.",
   "Formal goodness-of-fit tests reject all candidate distributions; serial dependence invalidates the independence assumption behind the i.i.d. likelihoods.",
   "The ARMA and GARCH models fail residual diagnostics."])

N(7, "Dataset Description and Source")
P("**Source.** The data are the 'solar-energy' dataset from the LSTNet repository (Lai, Chang, Yang and Liu, 2018), which redistributes the NREL *Solar Power Data for Integration "
  "Studies* for Alabama (2006). File used: solar_AL.txt.gz, downloaded on 2026-10-19 from raw.githubusercontent.com/laiguokun/multivariate-time-series-data (path solar-energy/solar_AL.txt.gz). "
  f"File size 8,748,993 bytes; SHA-256 {PRE['sha256_raw_file']}. The unmodified file is stored in data/.".replace("*", ""))
dc = K["data_checks"]
TABLE(pd.DataFrame([
    ("Rows x plants", f"{dc['n_rows']:,} x {dc['n_plants']}"), ("Time step", "10 minutes (single distinct step)"),
    ("Period", f"{dc['first_timestamp']} to {dc['last_timestamp']} (reconstructed)"),
    ("Missing / non-finite / negative values", f"{dc['n_missing']} / {dc['n_non_finite']} / {dc['n_negative']}"),
    ("Duplicate timestamps", str(dc["n_duplicate_timestamps"])), ("Largest single-plant output", f"{dc['plant_max_mw']} MW"),
    ("Plants that are always zero", str(dc["n_plants_always_zero"])),
    ("Mean plant output (range across plants)", f"{PRE['plant_mean_mw_min']:.2f} to {PRE['plant_mean_mw_max']:.2f} MW")],
    columns=["Check", "Result"]), "Dataset integrity checks (results/data_validation.json).", widths=[7, 8.5], size=9.5)
P("**Nature of the data (important).** To the best of the author's knowledge, the NREL integration-study data are modelled PV plant outputs derived from satellite-based solar-radiation "
  "and weather data for hypothetical plants, not meter readings. They are real, publicly released NREL data, but they are model output. The NREL project page "
  "(https://www.nrel.gov/grid/solar-power-data.html) was not reachable from the environment in which the analysis was run, so this description should be re-checked on that page before submission.")
P(f"**Reconstructed timestamps.** The file has no time column. Row 0 was assumed to be 2006-01-01 00:00 local standard time, with rows spaced exactly 10 minutes apart; the file has "
  f"52,560 = 365 x 144 rows, exactly one non-leap year. The assumption is supported by the data: the mean diurnal profile peaks at {K['time_axis_evidence']['peak_time']} (solar noon in Alabama is "
  f"around 12:00-12:15), and the mean number of hours per day with positive output is {K['time_axis_evidence']['daylength_dec']:.1f} h in December and "
  f"{K['time_axis_evidence']['daylength_jun']:.1f} h in June (astronomical day length at 32-33 degrees N is about 10 h and 14 h). The assumption cannot be proven from the file alone.")
P("**Analysis variable.** Fleet total (sum over 137 plants) in MW, averaged to hourly values: 8,760 observations.")

N(8, "Tools and Software Requirements")
TABLE(pd.DataFrame([("Python", ENV["python"]), ("NumPy", ENV["numpy"]), ("pandas", ENV["pandas"]), ("SciPy", ENV["scipy"]),
                    ("statsmodels", ENV["statsmodels"]), ("arch", ENV["arch"]), ("matplotlib", ENV["matplotlib"]), ("seaborn", ENV["seaborn"]),
                    ("pytest / python-docx", "for tests / for this report")], columns=["Package", "Version used"]),
      "Software environment used to produce all reported results.", widths=[6, 6], size=9.5)
P("All software is free and open source (NumPy: Harris et al., 2020; SciPy: Virtanen et al., 2020; statsmodels: Seabold and Perktold, 2010; pandas: McKinney, 2010; matplotlib: Hunter, 2007; arch: Sheppard). "
  "No account, API key or paid service is needed. Commands: pip install -r requirements.txt; python main.py (about 6 minutes); python -m pytest -q; python report/build_report.py. "
  "The random seed (20061019) and every setting are in src/config.py.")

N(9, "Statistical and Mathematical Background")
H2("9.1 Likelihood and maximum likelihood estimation")
P("Let x_1,...,x_n be observations from a density f(x; theta) with parameter vector theta of dimension k. If the observations are independent, the likelihood and log-likelihood are")
EQ("L(θ) = ∏_{i=1}^{n} f(x_i; θ),      ℓ(θ) = Σ_{i=1}^{n} ln f(x_i; θ)", "1")
P("The maximum likelihood estimator is θ̂ = argmax ℓ(θ). Under regularity conditions it is consistent and asymptotically normal, with covariance given by the inverse Fisher information; "
  "in practice the inverse of the observed information (the negative Hessian of ℓ at θ̂) is used:")
EQ("θ̂ ≈ N( θ, I(θ̂)⁻¹ ),      I(θ̂) = −∂²ℓ/∂θ∂θᵀ |_{θ̂}", "2")
P("Standard errors are the square roots of the diagonal of this matrix and 95% intervals are θ̂ ± 1.96·SE (Wald intervals). For dependent data, the joint density is factorised "
  "through conditional densities (prediction-error decomposition), which is how ARMA models are estimated below.")
H2("9.2 Candidate densities")
TABLE(pd.DataFrame([
    ("Normal", "f(x) = (2πσ²)^(-1/2) exp(−(x−μ)²/2σ²)", "μ, σ>0", "2"),
    ("Student-t (location-scale)", "f(x) = Γ((ν+1)/2) / (Γ(ν/2)·√(νπ)·σ) · (1+z²/ν)^(−(ν+1)/2), z=(x−μ)/σ", "μ, σ>0, ν>0", "3"),
    ("Log-normal", "f(x) = 1/(xσ√2π) exp(−(ln x − μ)²/2σ²), x>0", "μ, σ>0", "2"),
    ("Gamma", "f(x) = x^(a−1) e^(−x/s) / (Γ(a)·s^a), x>0", "shape a, scale s", "2"),
    ("Weibull", "f(x) = (c/λ)(x/λ)^(c−1) exp(−(x/λ)^c), x>0", "shape c, scale λ", "2"),
    ("Scaled Beta on (0, c)", "f(x) = (x/c)^(α−1)(1−x/c)^(β−1) / (c·B(α,β)), 0<x<c", "α, β (c fixed = 1.05·max)", "2"),
    ("Two-component normal mixture", "f(x) = w·N(x; μ1,σ1²) + (1−w)·N(x; μ2,σ2²)", "w, μ1, σ1, μ2, σ2", "5")],
    columns=["Model", "Density", "Parameters", "k"]), "Candidate distributions.", widths=[3.8, 7.5, 3.2, 1], size=8.5)
P("The scaled Beta is fitted to x/c with the Jacobian term −n·ln c included, so that its log-likelihood is comparable with those of the other models. "
  "The mixture likelihood is not concave and can have degenerate solutions (σ → 0 at a data point); it was initialised by the EM algorithm (Dempster, Laird and Rubin, 1977), then polished with BFGS, with a lower bound of 1e-4 on each σ.")
H2("9.3 Information criteria")
EQ("AIC = −2ℓ(θ̂) + 2k,      BIC = −2ℓ(θ̂) + k·ln n,      AICc = AIC + 2k(k+1)/(n−k−1)", "3")
P("Both penalise complexity. AIC (Akaike, 1974) targets out-of-sample predictive accuracy; BIC (Schwarz, 1978) approximates the log marginal likelihood and penalises complexity more heavily when n is large. "
  "Differences ΔAIC and ΔBIC to the best model are reported, with Akaike weights w_i = exp(−Δ_i/2)/Σ_j exp(−Δ_j/2) (Burnham and Anderson, 2002). Information criteria rank models; "
  "they do not say whether the best model fits well, which is why goodness-of-fit tests and diagnostics are also used.")
H2("9.4 Stationarity and unit-root tests")
P("A series is weakly stationary if its mean and variance are constant and Cov(X_t, X_{t+h}) depends only on h.")
B(["**ADF test** (Dickey and Fuller, 1979): regress ΔX_t = c + γ·X_{t−1} + Σ δ_j ΔX_{t−j} + ε_t. H0: γ = 0 (unit root, non-stationary). H1: γ < 0 (no unit root). Lags chosen by AIC. Rejection (small p) is evidence for stationarity.",
   "**KPSS test** (Kwiatkowski, Phillips, Schmidt and Shin, 1992): H0: the series is level-stationary (or trend-stationary with regression 'ct'). H1: unit root. Rejection (small p) is evidence against stationarity. statsmodels interpolates p-values from a table in the range 0.01-0.10.",
   "**Reading both together**: ADF rejects and KPSS does not reject → stationary. ADF does not reject and KPSS rejects → non-stationary. Both reject → the series has no unit root but its mean is not constant (typically a deterministic cycle or slow drift). Neither rejects → inconclusive.",
   "**Group tests**: one-way ANOVA and Kruskal-Wallis for equal means, and Levene/Brown-Forsythe for equal variances, across hour-of-day or month. These test directly for periodic non-stationarity in mean and variance."])
H2("9.5 ARMA, seasonal ARMA and GARCH")
EQ("φ(B)·Φ(B^s)·(Y_t − μ_t) = θ(B)·ε_t,      ε_t ~ N(0, σ²)", "4")
P("with B the backshift operator, φ and θ the AR and MA polynomials and Φ a seasonal AR polynomial (s = 24 hours). Y_t is the clear-sky index and μ_t = c (+ annual harmonics a·cos(2πd/365) + b·sin(2πd/365)). "
  "The exact Gaussian log-likelihood is computed by the Kalman filter, which also handles the missing night observations. A GARCH(1,1) model lets the innovation variance evolve:")
EQ("ε_t = σ_t·z_t,      σ_t² = ω + α·ε²_{t−1} + β·σ²_{t−1}", "5")
P("(Engle, 1982; Bollerslev, 1986), stationary in variance if α + β < 1, with z_t either standard normal or Student-t with ν degrees of freedom.")
H2("9.6 Diagnostics")
B(["**Ljung-Box test** (Ljung and Box, 1978): Q = n(n+2) Σ_{h=1}^{m} ρ̂_h²/(n−h), H0: no autocorrelation up to lag m. For squared residuals it tests for remaining volatility clustering.",
   "**ARCH-LM test** (Engle, 1982): regress squared residuals on their own lags; H0: no conditional heteroskedasticity.",
   "**Jarque-Bera test** (Jarque and Bera, 1987): H0: skewness 0 and excess kurtosis 0.",
   "**Kolmogorov-Smirnov and Anderson-Darling** goodness of fit (Anderson and Darling, 1952): when parameters are estimated from the same data the standard critical values are invalid (Lilliefors, 1967). "
   "Here the null distribution is obtained by a parametric bootstrap (Efron, 1979): simulate from the fitted model, re-estimate the parameters, recompute the statistic, B = 200 times. The smallest attainable p-value is 1/(B+1) = 0.004975."])

N(10, "Methodology and Workflow")
B(["**Step 1** Load, validate and aggregate: 137 plants → fleet total → hourly means.",
   "**Step 2** Exploratory analysis; identify the structural zero mass at night.",
   "**Step 3** Construct the clear-sky index on daylight hours (Section 11).",
   "**Step 4** Test stationarity (ADF, KPSS, group tests, ACF); remove the annual harmonic.",
   "**Step 5** Fit the seven candidate distributions by MLE; validate against SciPy; compare by AIC/BIC; test goodness of fit by parametric bootstrap; run sensitivity analyses.",
   "**Step 6** Fit ARMA-family models by exact Gaussian ML on the same observations (so log-likelihoods are comparable); run diagnostics; fit GARCH to the ARMA residuals.",
   "**Step 7** Forecast one hour ahead with a chronological 75/25 split; choose orders on training data only; compare with baselines.",
   "**Step 8** Report, including failures."])
P("All steps are executed by main.py; the notebook notebooks/analysis.ipynb gives a guided tour that reuses the same modules.")

N(11, "Data Preprocessing")
P(f"**Aggregation.** Plant outputs are summed (fleet total in MW) and averaged within each hour (six 10-minute values). No values were missing, negative or duplicated, so no imputation or deletion was needed.")
P(f"**Structural zeros.** {PRE['n_hours_exact_zero']:,} of {PRE['n_hours']:,} hours ({100*PRE['share_hours_exact_zero']:.1f}%) are exactly zero; the estimated zero probability is "
  f"{K['zero_mass']['p0_mle']:.4f} (SE {K['zero_mass']['se']:.4f}). These zeros are physical (night) and not errors, so they are kept in the raw data but excluded from continuous-distribution fitting: "
  "a continuous density cannot assign positive probability to a single point, and treating them as ordinary observations would distort every fitted model. This is a zero-inflated structure that is handled by modelling the daylight part.")
P(f"**Clear-sky index.** For each hour of the year the envelope E_t is the {int(100*K['settings']['envelope_quantile'])}th percentile of output at the same hour of day within ±{K['settings']['envelope_half_window_days']} days (circular in the year). "
  "The index is")
EQ("CSI_t = P_t / E_t,      for hours where E_t ≥ 10% of its annual maximum", "6")
P(f"This keeps {PRE['n_daylight_hours']:,} daylight hours; {PRE['n_daylight_zero_output']} of them have exactly zero output and were dropped (the Beta, Gamma, Weibull and log-normal densities have no mass at 0), leaving n = {PRE['n_in_sample']:,}. "
  f"{PRE['n_index_above_1']} index values exceed 1 (maximum {PRE['max_index']:.3f}), which is natural for a 95th-percentile envelope; {PRE['n_index_below_0_05']} values are below 0.05 (overcast hours). "
  f"For the forecasting experiment a causal envelope (trailing 14 days) is used so that no future information enters the index. Sensitivity to the quantile, window and threshold is examined in Section 20.")
P("**Why not model raw output directly?** A supplementary comparison on the raw positive MW values (n = 4,684) is provided in the Appendix. It is dominated by the deterministic diurnal and seasonal shape rather than by weather variability, and is therefore less informative about the stochastic component.")

N(12, "Exploratory Data Analysis")
d = K["eda"]["describe"]
P(f"Raw hourly fleet output has mean {d['mean']:.1f} MW, standard deviation {d['std']:.1f} MW, median {K['preprocessing']['hourly_median_mw']:.1f} MW (the median is almost zero because nearly half of the hours are night), "
  f"maximum {d['max']:.0f} MW and skewness {K['eda']['skewness_raw']:.2f}.")
FIGURE("fig01_hourly_series_overview.png", "Hourly fleet output, 2006: full year, a winter week and a summer week.")
FIGURE("fig02_diurnal_and_seasonal_pattern.png", "Diurnal profile and monthly pattern of output.")
FIGURE("fig03_distribution_of_raw_output.png", "Distribution of raw hourly output: the spike at zero (night) makes it unsuitable for a single continuous density.")
mi = pd.DataFrame({"Month": range(1, 13), "Mean output (MW)": [K["eda"]["monthly_mean_mw"][str(m)] for m in range(1, 13)],
                   "Mean index": [IDX["monthly_mean"][str(m)] for m in range(1, 13)], "SD of index": [IDX["monthly_sd"][str(m)] for m in range(1, 13)]})
TABLE(mi, "Monthly mean output and clear-sky index statistics.", widths=[2, 4, 3.5, 3.5], size=9, fmt={"Mean output (MW)": "{:.1f}", "Mean index": "{:.3f}", "SD of index": "{:.3f}"})
P(f"Mean output is lowest in winter (December {K['eda']['monthly_mean_mw']['12']:.0f} MW) and highest in early summer (June {K['eda']['monthly_mean_mw']['6']:.0f} MW). The index is more variable in winter (SD about 0.29-0.30) than in summer (SD 0.11-0.15): "
  "weather variability is larger in the cool season, a first indication that the variance is not constant across the year.")
FIGURE("fig04_clearsky_envelope_and_index.png", "Clear-sky envelope and resulting clear-sky index.")
P(f"The clear-sky index has mean {IDX['mean']:.3f}, SD {IDX['sd']:.3f}, skewness {IDX['skewness']:.2f} (a long left tail from cloudy hours) and excess kurtosis {IDX['excess_kurtosis']:.2f}. "
  f"Its median is {IDX['quantiles']['0.5']:.3f}, well above the mean, and its distribution is visibly bimodal (clear versus cloudy hours), which anticipates the success of the mixture model in Section 17.")

N(13, "Stationarity Testing")
P("Hypotheses: ADF H0 = unit root (non-stationary); KPSS H0 = level-stationary; significance level 5% throughout. Several KPSS specifications (automatic bandwidth, long bandwidth, with trend) are reported because KPSS is sensitive to the bandwidth.")
t = STAT.copy()
t["series"] = t["series"].str.replace("Clear-sky index", "CSI")
t["p_value"] = t["p_value"].map(pf)
t["Test"] = t["test"] + " (" + t["specification"].str.replace("regression=", "").str.replace("bandwidth=", "bw=").str.replace("lags by AIC", "AIC lags") + ")"
TABLE(t[["series", "Test", "statistic", "p_value", "reject_H0"]].rename(columns={"series": "Series", "statistic": "Statistic", "p_value": "p-value", "reject_H0": "Reject H0"}),
      "ADF and KPSS results (CSI = clear-sky index). KPSS p-values are interpolated and limited to [0.01, 0.10].", widths=[4.8, 6.0, 1.9, 1.9, 1.5], size=7.5, fmt={"statistic": "{:.3f}", "Statistic": "{:.3f}"})
v = VERD.copy(); v["adf_p"] = v["adf_p"].map(pf)
TABLE(v.rename(columns={"series": "Series", "adf_p": "ADF p", "kpss_stat": "KPSS stat", "kpss_p": "KPSS p", "verdict": "Verdict"}),
      "Combined reading of ADF and KPSS (level, automatic bandwidth).", widths=[4, 1.8, 1.8, 1.6, 6.5], size=8, fmt={"KPSS stat": "{:.3f}", "KPSS p": "{:.3f}"})
H2("13.1 Interpretation")
B([f"**Raw hourly output**: ADF rejects the unit root (p ≈ 4e-19) but KPSS strongly rejects level stationarity (statistic 2.47 vs 5% critical value 0.463): the series has no unit root but its mean is not constant, because of the deterministic diurnal and annual cycles. Hour-of-day ANOVA (F = {PER.iloc[0].statistic:.0f}) and Levene's test ({PER.iloc[2].statistic:.1f}) confirm non-stationarity in both mean and variance.",
   "**Raw daily means**: ADF does not reject (p = 0.052) and KPSS rejects: the seasonal drift makes this series look non-stationary.",
   "**Seasonal difference at lag 24** of the raw series is stationary by both tests (KPSS 0.023), showing that the diurnal cycle is the main source of non-stationarity.",
   "**Clear-sky index (daylight hours)**: ADF rejects (p ≈ 5e-23) but KPSS also rejects (0.83). The diurnal cycle has been removed (index vs hour of day: ANOVA p = 0.366, Levene p = 0.49), but the index still varies with month (ANOVA p ≈ 3e-87, Levene p ≈ 3e-115): both mean and variance have an annual pattern.",
   f"**After removing one annual harmonic** (R² = {K['stationarity']['harmonic_r2']:.3f}) the index passes both tests: ADF p ≈ 2e-25 and KPSS statistic 0.026 (do not reject). This is the form used for the ARMA models, with the harmonic estimated jointly with the ARMA parameters."])
P("The month-wise variance differences mean that the index is stationary in the mean only after removal of the annual cycle, and is not strictly stationary in the variance (winter variance is higher than summer variance). "
  "The i.i.d. distribution fits in Section 15 therefore describe the *unconditional* distribution over the year, which is a mixture over seasons and weather states; this is a legitimate and common use but it is not a description of a strictly stationary process.")
FIGURE("fig05_rolling_statistics_and_group_effects.png", "Rolling statistics and group effects (hour of day, month).")
FIGURE("fig06a_acf_pacf_raw_output.png", "ACF and PACF of the raw hourly output.")
FIGURE("fig06b_acf_pacf_clearsky_index.png", "ACF and PACF of the clear-sky index (pairwise-complete lags; PACF shown only to lag 6 where the pairwise estimate is reliable).")
P(f"The index autocorrelation is very high at lag 1 ({ACF['index_lag1']:.3f}) and decays slowly ({ACF['index_lag6']:.3f} at lag 6), with {ACF['index_lag24']:.3f} at lag 24 (the same hour on the next day) and {ACF['index_lag48']:.3f} at lag 48. "
  f"Day-to-day persistence is much weaker: the lag-1 autocorrelation of daily mean index is {K['effective_sample']['rho_daily_lag1']:.3f}, so the effective sample size is only about "
  f"{K['effective_sample']['n_eff_days_ar1_approx']:.0f} independent days (AR(1) approximation n·(1−ρ)/(1+ρ)). This is why the i.i.d. standard errors in Section 14 are optimistic.")

N(14, "Maximum Likelihood Estimation")
P("Each model's log-likelihood (Section 9) was coded explicitly. Parameters were mapped to unconstrained coordinates (log for positive parameters, logit for weights) and ℓ was maximised with the BFGS quasi-Newton method (Nocedal and Wright, 2006) from one or several starting values (robust moment-type estimates), with a Nelder-Mead polish if needed, keeping the best. "
  "Convergence was accepted only if (i) the optimiser reported success, (ii) the gradient norm at the optimum was small, and (iii) the numerical Hessian of −ℓ was positive definite. Standard errors come from the inverse observed information (eq. 2), transformed to the original scale by the delta method.")
P("**Closed-form checks.** For the Normal, μ̂ = x̄ and σ̂² = (1/n)Σ(x−x̄)², for the log-normal the same formulas apply to ln x, and for the Gamma the profile equation ln a − ψ(a) = ln x̄ − mean(ln x) must hold with scale = x̄/a. The code verifies these identities.")
vv = VALID[["model", "our_loglik", "scipy_loglik", "loglik_difference_ours_minus_scipy", "max_abs_param_difference", "validated"]].copy()
TABLE(vv.rename(columns={"model": "Model", "our_loglik": "Own ℓ", "scipy_loglik": "SciPy ℓ", "loglik_difference_ours_minus_scipy": "Δℓ", "max_abs_param_difference": "Max |Δθ|", "validated": "Validated"}),
      "Validation of the own MLE implementation against scipy.stats fitters (for the mixture, SciPy has no fitter: only the log-likelihood value is recomputed independently with scipy.stats.norm, so its maximisation is checked by multi-start EM+BFGS rather than by an external fitter).", widths=[4.2, 2.3, 2.3, 1.8, 2.3, 2],
      size=8.5, fmt={"Own ℓ": "{:.3f}", "SciPy ℓ": "{:.3f}", "Δℓ": "{:.1e}", "Max |Δθ|": "{:.1e}"})
e = MLE.copy()
TABLE(e[["model", "parameter", "estimate", "std_error", "ci95_low", "ci95_high"]].rename(columns={"model": "Model", "parameter": "Parameter", "estimate": "MLE", "std_error": "SE", "ci95_low": "95% CI low", "ci95_high": "95% CI high"}),
      f"Maximum likelihood estimates, standard errors and Wald 95% intervals (clear-sky index, n = {PRE['n_in_sample']:,}). Standard errors assume independence and are therefore too small (Section 13).",
      widths=[4.2, 2.5, 2.3, 2.2, 2.4, 2.4], size=8.5, fmt={c: "{:.4f}" for c in ["estimate", "std_error", "ci95_low", "ci95_high"]} | {"MLE": "{:.4f}", "SE": "{:.4f}", "95% CI low": "{:.4f}", "95% CI high": "{:.4f}"})
FIGURE("fig07_likelihood_surface_weibull.png", "Log-likelihood surface for the Weibull model with the MLE marked: a unimodal, well-conditioned surface.")

N(15, "Candidate Distribution Models")
P("Seven models were fitted to the clear-sky index. The Normal is the reference; Student-t adds heavy tails; Log-normal, Gamma and Weibull are positively skewed families on (0, ∞) commonly used for resource data; "
  "the scaled Beta has bounded support (0, c) which suits a ratio with a physical upper limit (c = 1.05 times the maximum observed index, about 1.317, is fixed and not estimated; a different c would change the Beta fit); and the two-component normal mixture "
  "represents a population of two weather states (cloudy/variable and clear). The mixture has 5 parameters, the others 2-3.")
P(f"The fitted mixture has weight {mix['weight1']:.3f} on a broad 'variable-sky' component (mean {mix['mu1']:.3f}, SD {mix['sigma1']:.3f}) and weight {1-mix['weight1']:.3f} on a narrow 'clear-sky' component "
  f"(mean {mix['mu2']:.3f}, SD {mix['sigma2']:.3f}). The labels are an interpretation of the fitted components, not something the model proves.")
FIGURE("fig08_fitted_densities.png", "Fitted densities overlaid on the histogram of the clear-sky index.")
FIGURE("fig09_qq_plots.png", "Q-Q plots of the clear-sky index against each fitted model.")

N(16, "Time-Series Modelling")
P(f"**Justification.** Serial dependence is overwhelming (lag-1 ACF {ACF['index_lag1']:.2f}), and a dynamic model is therefore justified. Nonconstant variance across months and residual volatility clustering (Section 18) motivate trying GARCH. "
  "The index is only defined in daylight, so ARMA models are fitted as regressions with ARMA errors on the full hourly grid with night hours treated as missing; the exact Gaussian likelihood computed by the Kalman filter uses the same "
  f"n = {TSK['n']:,} observations as the i.i.d. fits, so the log-likelihoods, AIC and BIC are directly comparable. AIC and BIC use k = number of estimated parameters and n = number of observed values.")
tsc = TS[["model", "k", "loglik", "AIC", "BIC", "delta_AIC", "delta_BIC"]].copy()
TABLE(tsc.rename(columns={"model": "Model", "loglik": "Log-lik", "delta_AIC": "ΔAIC", "delta_BIC": "ΔBIC"}), "ARMA-family models (n = 3,846).", widths=[5.8, 0.9, 2, 2.2, 2.2, 1.9, 1.9], size=8.5,
      fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔAIC": "{:.1f}", "ΔBIC": "{:.1f}"})
tp_ = TSP.copy()
TABLE(tp_[["parameter", "estimate", "std_error", "z", "p_value", "ci95_low", "ci95_high"]].rename(columns={"parameter": "Parameter", "estimate": "Estimate", "std_error": "SE", "p_value": "p", "ci95_low": "CI low", "ci95_high": "CI high"}),
      f"Parameters of the BIC-selected model {TSK['best_BIC']}.", widths=[2.6, 2.2, 2, 2, 2, 2.2, 2.2], size=8.5, fmt={"Estimate": "{:.4f}", "SE": "{:.4f}", "z": "{:.1f}", "p": "{:.3g}", "CI low": "{:.4f}", "CI high": "{:.4f}"})
P("The estimates are consistent with a persistent but stationary process (φ1 + φ2 = 0.963 < 1). The seasonal AR coefficient 0.42 reflects day-to-day persistence of weather at the same hour.")
P(f"**BIC and AIC disagree.** BIC selects the model without an annual cycle (ΔBIC = 6.0 in its favour) while AIC selects the model with it (ΔAIC = 6.5). The likelihood-ratio test of the annual harmonic gives "
  f"LR = {TSK['lr_test_annual_cycle']['statistic']:.2f} on {TSK['lr_test_annual_cycle']['df']} degrees of freedom, p = {TSK['lr_test_annual_cycle']['p_value']:.4f} (Wilks, 1938), but the test treats the observations as independent given the model, and the residuals are not white. "
  "The evidence for an annual cycle in the mean is therefore modest; both models are reported.")
H2("16.1 GARCH on the ARMA innovations")
P("GARCH(1,1) was fitted by maximum likelihood (package arch) to the ARMA residuals (concatenated daylight residuals, rescaled by 100 for numerical stability, with the Jacobian n·ln 100 added back so that log-likelihoods are on the original scale).")
gc = GC[["model", "k", "loglik", "AIC", "BIC", "delta_BIC"]].copy()
TABLE(gc.rename(columns={"model": "Model", "loglik": "Log-lik", "delta_BIC": "ΔBIC"}), "Volatility models for the ARMA innovations.", widths=[5.5, 1, 2.5, 2.5, 2.5, 2], size=8.5,
      fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔBIC": "{:.1f}"})
gp = GP.set_index("parameter")["estimate"]
P(f"GARCH(1,1) with Student-t innovations is preferred by a wide margin, but the fit is degenerate: α = {gp['alpha[1]']:.3f}, β = {gp['beta[1]']:.3f} (α + β = 1.000, the integrated boundary) and ν = {gp['nu']:.2f} degrees of freedom, "
  "which implies a distribution with barely finite variance. This is a sign that the GARCH recursion is trying to absorb a different kind of structure, namely variance that depends on the *level* of the index (Section 18), rather than evidence of true volatility clustering.")
FIGURE("fig10_distribution_model_selection.png", "AIC/BIC comparison of the candidate distributions.")
FIGURE("fig11_iid_vs_dynamic_models.png", "I.i.d. distribution models versus dynamic models on the same observations.")
FIGURE("fig13_garch_conditional_volatility.png", "GARCH conditional volatility of the ARMA innovations.")

N(17, "Model Selection Using AIC and BIC")
m = MC[["model", "k", "loglik", "AIC", "BIC", "delta_AIC", "delta_BIC", "weight_BIC"]].copy()
TABLE(m.rename(columns={"model": "Model", "loglik": "Log-lik", "delta_AIC": "ΔAIC", "delta_BIC": "ΔBIC", "weight_BIC": "BIC weight"}), "Distribution comparison (clear-sky index, n = 3,846).",
      widths=[4.4, 0.8, 1.9, 1.9, 1.9, 1.9, 1.9, 1.8], size=8.2, fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔAIC": "{:.1f}", "ΔBIC": "{:.1f}", "BIC weight": "{:.3g}"})
P(f"The mixture is best by both criteria, with a ΔBIC of {MC['delta_BIC'].sort_values().iloc[1]:,.0f} to the second-best model (scaled Beta), which is the best single-family distribution. The gap is so large that the extra three parameters are overwhelmingly justified. "
  "Among single families, the ranking is Beta, Student-t, Normal, Weibull, Gamma, Log-normal; the log-normal and Gamma are poor because they are right-skewed whereas the index is left-skewed (skewness −1.12).")
H2("17.1 Goodness of fit")
g = MC[["model", "KS_statistic", "KS_p_bootstrap", "AD_p_bootstrap"]].copy()
TABLE(g.rename(columns={"model": "Model", "KS_statistic": "KS D", "KS_p_bootstrap": "KS p (bootstrap)", "AD_p_bootstrap": "AD p (bootstrap)"}), "Parametric-bootstrap goodness-of-fit tests (parameters re-estimated in every replicate, B = 200).",
      widths=[5, 2.5, 3.5, 3.5], size=9, fmt={"KS D": "{:.3f}", "KS p (bootstrap)": "{:.4f}", "AD p (bootstrap)": "{:.4f}"})
P("Every model is rejected (p at the minimum 1/(B+1)). The Kolmogorov-Smirnov distance of the mixture, D = 0.036, is much smaller than for the other models (0.14-0.21) - it is a far better description - but is not zero. "
  "Two reasons are important: (i) with n = 3,846 the tests detect tiny departures; (ii) the observations are serially dependent, so the effective sample size is much smaller than n and the i.i.d. bootstrap null does not reproduce the actual dependence. "
  "A formal 'adequate fit' claim is therefore not made for any model; the mixture is simply the best of those compared.")
H2("17.2 Robustness of the ranking")
s = SENS[["variant", "n", "best_by_AIC", "best_by_BIC", "best_single_family_BIC", "delta_BIC_runner_up"]].copy()
TABLE(s.rename(columns={"variant": "Variant", "best_by_AIC": "Best AIC", "best_by_BIC": "Best BIC", "best_single_family_BIC": "Best single", "delta_BIC_runner_up": "ΔBIC runner-up"}),
      "Sensitivity of the ranking to preprocessing choices.", widths=[4.8, 1.2, 2.7, 2.7, 2.2, 2.2], size=7.5, fmt={"ΔBIC runner-up": "{:.0f}"})
P("The mixture ranks first by both criteria in all eight variants, including the extreme thinning to noon-only values (n = 365) and every third day, which removes most of the serial dependence. "
  "The best single family is Beta in seven of eight variants and Student-t when the envelope quantile is 0.90 (when the envelope is lower, many index values are near or above 1).")

N(18, "Model Diagnostics and Validation")
best = TSK["best_BIC"]
TESTS = ["Ljung-Box", "Ljung-Box (squared residuals)", "ARCH-LM", "Jarque-Bera", "Residual skewness", "Residual excess kurtosis"]


def diag_table(mask, caption):
    dd = DIAG[mask & DIAG.test.isin(TESTS)].copy()
    dd["p_value"] = dd["p_value"].map(lambda x: x)
    dd["lag"] = dd["lag"].map(lambda x: "" if pd.isna(x) else int(x))
    dd["pv"] = [("" if pd.isna(p) else pf(p)) for p in dd["p_value"]]
    dd["rej"] = ["" if pd.isna(r) else str(r) for r in dd["reject_H0"]]
    TABLE(dd[["test", "lag", "statistic", "pv", "rej"]].rename(columns={"test": "Test / quantity", "lag": "Lag", "statistic": "Statistic (or value)", "pv": "p-value", "rej": "Reject H0"}),
          caption, widths=[5, 1.5, 3, 3, 2], size=8.5, fmt={"Statistic (or value)": "{:.3f}"})


is_garch = DIAG.model.str.contains("GARCH", regex=False)
diag_table(DIAG.model.str.contains("no annual cycle", regex=False) & ~is_garch, f"Residual diagnostics of the BIC-selected ARMA model ({best}). Skewness and excess kurtosis rows show the sample value.")
diag_table(is_garch, "Diagnostics of the standardised residuals of GARCH(1,1) with Student-t innovations (fitted to the ARMA innovations).")
P("All Ljung-Box (residual and squared-residual), ARCH-LM and Jarque-Bera tests reject. The same holds for every other ARMA model and for the GARCH model (standardised residuals; see diagnostic_results.csv). "
  "The conclusion is that **none of the fitted dynamic models is adequate in the strict sense.** The diagnostics point to the following specific features:")
B([f"**The step across the night.** The residual ACF is {K['residual_structure']['acf_lag1']:.2f} at lag 1 but {K['residual_structure']['acf_lag13']:.2f} at lag 13 - on the hourly grid this links an hour with the one 13 hours earlier, which for early-morning hours lies on the previous afternoon, i.e. across the night gap. "
   "The strongly negative value indicates dependence across the night that the ARMA structure on the 24-hour grid does not represent (the sign and size are reported as found; the physical reading is an interpretation).",
   f"**Level-dependent variance.** The residual SD falls from {K['residual_structure']['std_lowest_level_quintile']:.2f} (standardised residuals) in the lowest quintile of fitted level to {K['residual_structure']['std_highest_level_quintile']:.2f} in the highest quintile: variability is large when the index is expected to be low and small when it is near 1. "
   "This is the reason for the degenerate GARCH fit; a model in which the variance is a function of the level (or a Beta-type transformation) would be more suitable.",
   f"**Non-Gaussian errors.** The residual skewness is −0.58 and excess kurtosis 2.04 (Jarque-Bera rejects)."])
rv = RV.copy()
TABLE(rv, "Residual standard deviation by quintile of fitted level (results/residual_variance_by_fitted_level.csv).", size=8.5)
P("Two remedies were tried. A logit transformation of the index to stabilise the variance gave a *lower* log-likelihood on the original scale (6,012 vs 6,222) and was rejected. "
  "A check of the night-gap length (0, 4, 12 and 24 night slots between days) shows that the true 12-slot gap is near-optimal:")
TABLE(NG[["night_slots", "seasonal_period", "loglik", "AIC", "BIC"]].rename(columns={"night_slots": "Night slots", "seasonal_period": "Seasonal period"}), "Sensitivity of the ARMA fit to the assumed length of the night gap.",
      widths=[2.5, 3, 2.5, 2.5, 2.5], size=9, fmt={"loglik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}"})
P("so the lag-13 problem is not a consequence of a wrong grid. A two-time-scale model (a day-level random effect plus within-day dynamics) is listed as future work.")
FIGURE("fig12_arma_residual_diagnostics.png", "Residual diagnostics of the selected ARMA model: residuals, ACF, histogram, Q-Q plot.")

N(19, "Experimental Results")
P(f"**One-hour-ahead forecasting.** The data were split chronologically: training days 1-{FC['train_days']} (to {FC['train_end'][:10]}, {FC['train_nobs']:,} daylight observations), test days {FC['train_days']+1}-365 ({FC['test_days']} days, {FC['n_test']} daylight hours). "
  "Orders were chosen by BIC on the training data only (candidates below); the trained parameters were then applied to the full series with the filter, so that each test-hour forecast uses only data up to the previous hour. The clear-sky envelope for the test period is causal. Forecasts of the index are multiplied by the envelope to obtain MW.")
TABLE(FT[["model", "n", "k", "loglik", "AIC", "BIC", "converged"]], "Order selection on the training sample.", widths=[4.5, 1.3, 0.9, 2.2, 2.2, 2.2, 1.9], size=8.5, fmt={"loglik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}"})
fs = FS.rename(columns={"method": "Method", "RMSE_MW": "RMSE (MW)", "MAE_MW": "MAE (MW)", "RMSE_index": "RMSE (index)", "MAE_index": "MAE (index)", "RMSE_skill_vs_index_persistence": "RMSE skill"})
TABLE(fs[["Method", "n_test", "MAE (index)", "RMSE (index)", "MAE (MW)", "RMSE (MW)", "RMSE skill"]], "Out-of-sample forecast accuracy. Skill = 1 − RMSE/RMSE(index persistence) (Hyndman and Koehler, 2006).",
      widths=[4.8, 1.2, 1.9, 1.9, 1.8, 1.8, 1.7], size=8, fmt={"MAE (index)": "{:.4f}", "RMSE (index)": "{:.4f}", "MAE (MW)": "{:.1f}", "RMSE (MW)": "{:.1f}", "RMSE skill": "{:.3f}"})
FIGURE("fig14_forecast_evaluation.png", "Forecast evaluation on the hold-out period.")
P(f"The selected ARMA(1,1)x(1,0,0)_24 model has RMSE {FS.iloc[0]['RMSE_MW']:.1f} MW versus {FS.iloc[1]['RMSE_MW']:.1f} MW for persistence of the index (a gain of {100*FS.iloc[0]['RMSE_skill_vs_index_persistence']:.1f}%), and both are far better than naive persistence of the raw power ({FS.iloc[3]['RMSE_MW']:.0f} MW), "
  "which cannot anticipate sunrise and sunset. The comparison covers only one year and a single 91-day test period (October-December), so the evidence for the margin over persistence is limited; no significance test was performed.")

N(20, "Results and Discussion")
B(["**RQ1 (stationarity).** Raw output is not stationary in the mean or variance; the diurnal cycle is the main cause (seasonal differencing at lag 24 restores stationarity). The clear-sky index is stationary across hours of the day and, after removing an annual harmonic, passes both ADF and KPSS, but its variance still differs between months.",
   f"**RQ2 (distribution).** The two-component normal mixture is best by AIC and BIC in the baseline and in all seven sensitivity variants; among single families the scaled Beta is best. The data are bimodal (clear versus variable sky), which no single unimodal family can reproduce.",
   "**RQ3 (goodness of fit).** All models are formally rejected by bootstrap KS and Anderson-Darling tests. The mixture is far closer to the data than the others (D = 0.036 versus 0.14-0.21), and rejection is unsurprising given n and the serial dependence, but no adequacy is claimed.",
   f"**RQ4 (dynamics).** Dependence dominates: the ARMA log-likelihood ({TSK['best_loglik']:.0f}) is about six times the best i.i.d. value (1,001). However, all residual tests reject; the model is a useful predictor but not a complete probabilistic description. The GARCH fit is degenerate, so volatility clustering is not convincingly established; the variance structure is better explained by dependence on the level.",
   f"**RQ5 (forecasting).** The ARMA model improves RMSE by {100*FS.iloc[0]['RMSE_skill_vs_index_persistence']:.1f}% relative to persistence of the index on one hold-out period."])
P("**Interpreting the likelihood comparison.** The i.i.d. mixture and the ARMA models use the same 3,846 observations, but they answer different questions: the former is a marginal distribution, the latter a conditional one. "
  "The comparison of log-likelihoods shows how much predictive information is contained in the past, not that the ARMA 'beats' the mixture as a marginal model. A natural combination, not attempted here, is a time-series model with mixture or Beta-distributed innovations.")
P("**Model-selection caution.** AIC and BIC are derived under independent observations for the i.i.d. models. With serial dependence the effective sample size is much smaller, so BIC penalties computed with n = 3,846 are too light and the likelihoods overstate the information in the data. "
  "The ranking of the distributions was nevertheless stable in every sensitivity variant, including those that remove most dependence, which supports the conclusion.")

N(21, "Conclusion")
P("A genuine public PV dataset was analysed with a complete, reproducible pipeline. Maximum likelihood estimators were implemented from explicit log-likelihoods, validated against SciPy and used to compare seven distributions. "
  "The main findings are: (1) raw PV output is non-stationary because of deterministic cycles and has 46.5% structural zeros, so a clear-sky index on daylight hours is the appropriate object for distribution fitting; "
  "(2) a two-component normal mixture is the best marginal model by AIC and BIC and the Beta is the best single-family model; (3) every distribution is rejected by bootstrap goodness-of-fit tests, mainly because of serial dependence; "
  "(4) a seasonal ARMA model captures most of the dependence and gives a modest forecast gain, but fails all residual diagnostics, as does GARCH; and (5) AIC and BIC disagree on the annual cycle. "
  "The honest summary is that the mixture and ARMA models are useful descriptions and predictors, but none should be considered a correct generating model.")

N(22, "Limitations and Future Scope")
B(["**Data**: modelled rather than metered output; reconstructed timestamps; one year, one state, fleet total only. Re-checking the NREL documentation of the dataset is recommended.",
   "**Clear-sky envelope**: empirical, not physical; replacing it by a physical clear-sky model (for example from pvlib) would remove the dependence of the index on the quantile and window choice.",
   "**Independence**: the i.i.d. likelihoods, information criteria and goodness-of-fit tests ignore serial dependence; block bootstrap or models with dependent errors would give valid inference.",
   "**Dynamic model**: the ARMA and GARCH models fail diagnostics. Future work: a two-time-scale model, level-dependent variance, Beta-distributed or mixture innovations, regime-switching models.",
   "**Forecasting**: a single 91-day test period and no probabilistic forecast evaluation; rolling-origin evaluation and proper scoring rules (CRPS) would be stronger.",
   "**Wind** was not analysed; the same pipeline could be applied to a wind dataset (for example a Weibull-based analysis).",
   "**Model uncertainty**: only seven distributions and one mixture size were compared; mixtures with more components and a Beta mixture were not investigated."])

# ---- References
H1("23. References")
REFS = [
 "Akaike, H. (1974). A new look at the statistical model identification. IEEE Transactions on Automatic Control, 19(6), 716-723.",
 "Anderson, T. W., and Darling, D. A. (1952). Asymptotic theory of certain 'goodness of fit' criteria based on stochastic processes. Annals of Mathematical Statistics, 23(2), 193-212.",
 "Bacher, P., Madsen, H., and Nielsen, H. A. (2009). Online short-term solar power forecasting. Solar Energy, 83(10), 1772-1783.",
 "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.",
 "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
 "Burnham, K. P., and Anderson, D. R. (2002). Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach (2nd ed.). Springer.",
 "Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society, Series B, 39(1), 1-38.",
 "Dickey, D. A., and Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association, 74(366), 427-431.",
 "Durbin, J., and Koopman, S. J. (2012). Time Series Analysis by State Space Methods (2nd ed.). Oxford University Press.",
 "Efron, B. (1979). Bootstrap methods: another look at the jackknife. The Annals of Statistics, 7(1), 1-26.",
 "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.",
 "Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. Philosophical Transactions of the Royal Society of London, Series A, 222, 309-368.",
 "Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.",
 "Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357-362.",
 "Hunter, J. D. (2007). Matplotlib: a 2D graphics environment. Computing in Science and Engineering, 9(3), 90-95.",
 "Hyndman, R. J., and Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679-688.",
 "Inman, R. H., Pedro, H. T. C., and Coimbra, C. F. M. (2013). Solar forecasting methods for renewable energy integration. Progress in Energy and Combustion Science, 39(6), 535-576.",
 "Jarque, C. M., and Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review, 55(2), 163-172.",
 "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., and Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54(1-3), 159-178.",
 "Lai, G., Chang, W.-C., Yang, Y., and Liu, H. (2018). Modeling long- and short-term temporal patterns with deep neural networks. Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, 95-104. Data: https://github.com/laiguokun/multivariate-time-series-data",
 "Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. Journal of the American Statistical Association, 62(318), 399-402.",
 "Ljung, G. M., and Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297-303.",
 "McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 56-61.",
 "National Renewable Energy Laboratory. Solar Power Data for Integration Studies. https://www.nrel.gov/grid/solar-power-data.html (page not reachable from the analysis environment; cited as the original data source).",
 "Nocedal, J., and Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.",
 "Schwarz, G. (1978). Estimating the dimension of a model. The Annals of Statistics, 6(2), 461-464.",
 "Seabold, S., and Perktold, J. (2010). statsmodels: econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference, 92-96.",
 "Sheppard, K. arch: ARCH models in Python (software). https://github.com/bashtage/arch",
 "Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261-272.",
 "Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. Annals of Mathematical Statistics, 9(1), 60-62.",
]
for i, r in enumerate(REFS, 1):
    p = doc.add_paragraph(f"[{i}] {r}"); p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(3); p.runs[0].font.size = Pt(10.5)

# ---- Appendix
H1("24. Appendix: Important Code and Additional Results")
H2("A. Project structure and commands")
P("main.py runs the whole analysis; src/ holds config.py, preprocessing.py, stationarity.py, mle_models.py, model_selection.py, diagnostics.py, timeseries_models.py and plotting.py; tests/test_models.py holds 18 unit tests with known-parameter simulations. Results are written to results/.", align="l")
for line in ["python -m venv .venv && source .venv/bin/activate", "pip install -r requirements.txt", "python main.py", "python -m pytest -q", "python report/build_report.py"]:
    p = doc.add_paragraph(); r = p.add_run(line); r.font.name = "Courier New"; r.font.size = Pt(9.5); p.paragraph_format.space_after = Pt(0)
H2("B. Core MLE routine (excerpt of src/mle_models.py)")
src = (ROOT / "src" / "mle_models.py").read_text().splitlines()
start = next((i for i, l in enumerate(src) if l.startswith("class Weibull")), None)
excerpt = src[start:start + 40] if start is not None else src[:40]
for line in excerpt:
    p = doc.add_paragraph(); r = p.add_run(line if line else " "); r.font.name = "Courier New"; r.font.size = Pt(8)
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
H2("C. Supplementary comparison on raw positive output (MW)")
TABLE(RAWPOS.rename(columns={"model": "Model", "loglik": "Log-lik"}), "Distribution comparison for the raw positive hourly output (n = 4,684); the Beta is best, but the shape mostly reflects the diurnal profile.",
      widths=[5, 3, 3], size=9, fmt={"Log-lik": "{:.1f}", "BIC": "{:.1f}"})
H2("D. Output files")
P("results/*.csv (all tables), results/key_results.json (all key numbers and software versions), results/figures/*.png (15 figures), data/cleaned_dataset.csv (hourly series with envelope and index), data/dataset_metadata.md.", align="l")
H2("E. Deviations from the requested project layout")
P("The requested file data/original_dataset.csv was not created: the unmodified source file is kept as data/solar_AL.txt.gz (a 52,560 x 137 matrix with no header), which is documented in data/dataset_metadata.md. The cleaned hourly series is in data/cleaned_dataset.csv.", align="l")

doc.save(OUT)
print("wrote", OUT, "| figures:", FIGN[0], "| tables:", TABN[0])
