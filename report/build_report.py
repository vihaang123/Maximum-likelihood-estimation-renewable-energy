"""Builds report/project_report.docx in the compact course-report format (title block, Abstract, numbered
sections 1-6, References, 'Page X of Y' footer).  All numbers are read from results/*.csv and key_results.json.

Run from the project root:   python report/build_report.py
(The longer 27-part version is built by report/build_report_detailed.py.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
OUT = Path(__file__).resolve().parent / "project_report.docx"
TITLE = "Maximum Likelihood Estimation and Model Selection for Stationary Renewable Energy Output Data"
NAVY, BLUE, GREY = RGBColor(0x1F, 0x38, 0x64), RGBColor(0x2F, 0x54, 0x96), RGBColor(0x59, 0x59, 0x59)
FONT = "Calibri"

K = json.loads((RES / "key_results.json").read_text())
rd = lambda n: pd.read_csv(RES / n)
STAT, VERD, MLE, MC, SENS = rd("stationarity_tests.csv"), rd("stationarity_verdicts.csv"), rd("mle_estimates.csv"), rd("model_comparison.csv"), rd("sensitivity_distribution_ranking.csv")
TS, TSP, DIAG, GC, GP = rd("timeseries_model_comparison.csv"), rd("timeseries_best_model_parameters.csv"), rd("diagnostic_results.csv"), rd("garch_comparison.csv"), rd("garch_best_model_parameters.csv")
FS, FT, VALID, PER = rd("forecast_scores.csv"), rd("forecast_training_model_selection.csv"), rd("mle_validation.csv"), rd("periodicity_tests.csv")
PRE, IDX, ACF, TSK, FC, ENV, RS = K["preprocessing"], K["index_summary"], K["acf"], K["timeseries"], K["forecast"], K["environment"], K["residual_structure"]
NG = rd("night_gap_sensitivity.csv")
mix = MLE[MLE.model == "2-component normal mixture"].set_index("parameter")["estimate"]
mcs = MC.set_index("model")
runner_up = mcs["delta_BIC"].sort_values().iloc[1]
gp = GP.set_index("parameter")["estimate"]
skill = 100 * FS.iloc[0]["RMSE_skill_vs_index_persistence"]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.54)
sec.top_margin, sec.bottom_margin = Cm(2.3), Cm(2.2)

st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after, st.paragraph_format.line_spacing = Pt(6), 1.15
for name, size, color, italic in (("Heading 1", 14, NAVY, False), ("Heading 2", 12, BLUE, False), ("Heading 3", 10.5, BLUE, True)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold, h.font.italic, h.font.color.rgb = FONT, Pt(size), True, italic, color
    h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); h.element.rPr.rFonts.set(qn("w:ascii"), FONT); h.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    h.paragraph_format.space_before, h.paragraph_format.space_after, h.paragraph_format.keep_with_next = Pt(12 if name == "Heading 1" else 9), Pt(4), True

FIGN, TABN = [0], [0]


def runs(p, text, size=None, italic=False, color=None):
    for i, part in enumerate(text.split("**")):
        if part:
            r = p.add_run(part); r.bold = i % 2 == 1; r.italic = italic
            if size: r.font.size = Pt(size)
            if color: r.font.color.rgb = color


def P(text, align="j", after=None, **kw):
    p = doc.add_paragraph(); runs(p, text, **kw)
    p.alignment = {"j": WD_ALIGN_PARAGRAPH.JUSTIFY, "c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT}[align]
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p


def LIST(items, numbered=True):
    for t in items:
        p = doc.add_paragraph(style="List Number" if numbered else "List Bullet"); runs(p, t)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(2)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(6)


def restart_numbering():
    """Give the next numbered list its own numbering instance so it starts again at 1."""
    numbering = doc.part.numbering_part.numbering_definitions._numbering
    base = doc.styles["List Number"].element.pPr.numPr.numId.val
    abstract = [n for n in numbering.findall(qn("w:num")) if n.get(qn("w:numId")) == str(base)][0].find(qn("w:abstractNumId")).get(qn("w:val"))
    num = numbering.add_num(int(abstract)); num.add_lvlOverride(ilvl=0).add_startOverride(1)
    return num.numId


def NLIST(items):
    nid = restart_numbering()
    for t in items:
        p = doc.add_paragraph(style="List Number"); runs(p, t)
        pPr = p._p.get_or_add_pPr(); numPr = OxmlElement("w:numPr")
        il = OxmlElement("w:ilvl"); il.set(qn("w:val"), "0"); ni = OxmlElement("w:numId"); ni.set(qn("w:val"), str(nid))
        numPr.append(il); numPr.append(ni); pPr.append(numPr)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(2)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(6)


def EQ(text, label=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.italic = True; r.font.name = "Cambria Math"
    if label: p.add_run(f"      ({label})").font.size = Pt(9.5)
    p.paragraph_format.space_after = Pt(5)


def FIGURE(fname, caption, width=15.2):
    FIGN[0] += 1
    doc.add_picture(str(FIG / fname), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Figure {FIGN[0]}: {caption}"); r.italic = True; r.font.size = Pt(9.5); r.font.color.rgb = GREY
    p.paragraph_format.space_after = Pt(8)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color); tcPr.append(shd)


def borders(table, color="BFBFBF"):
    tblPr = table._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{e}"); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), color); b.append(el)
    tblPr.append(b)


def TABLE(df, caption, widths=None, size=8.5, fmt=None, bold_rows=(), left_cols=(0,)):
    TABN[0] += 1
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Table {TABN[0]}: {caption}"); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = NAVY
    cap.paragraph_format.keep_with_next = True; cap.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(df.columns)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; borders(t)
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = ""; shade(cell, "1F3864")
        para = cell.paragraphs[0]; para.alignment = WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_after = Pt(0)
        run = para.add_run(str(c)); run.bold = True; run.font.size = Pt(size); run.font.color.rgb = RGBColor(255, 255, 255)
    for i, (_, row) in enumerate(df.iterrows()):
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            v = row[c]
            if isinstance(v, float) and pd.isna(v): s = "–"
            elif isinstance(v, float): s = (fmt or {}).get(c, "{:.4g}").format(v)
            else: s = str(v)
            cells[j].text = ""
            para = cells[j].paragraphs[0]; para.paragraph_format.space_after = Pt(0); para.paragraph_format.line_spacing = 1.0
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in left_cols else WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(s); run.font.size = Pt(size); run.bold = i in bold_rows
            if i in bold_rows: shade(cells[j], "FDE9E7")
    for row in t.rows:                                   # keep rows together, add a little padding
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
    if widths:
        t.autofit = False
        for j, w in enumerate(widths): t.columns[j].width = Cm(w)
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pf(p): return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def field(paragraph, instr, size=9):
    run = paragraph.add_run(); run.font.size = Pt(size)
    for kind in ("begin", "instr", "separate", "text", "end"):
        if kind == "instr":
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = instr
        elif kind == "text":
            el = OxmlElement("w:t"); el.text = "1"
        else:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Page "); r.font.size = Pt(8.5); r.font.color.rgb = GREY
field(fp, "PAGE"); r = fp.add_run(" of "); r.font.size = Pt(10)
field(fp, "NUMPAGES")

# ================================================================== TITLE BLOCK
p = P(TITLE, "c", after=4); p.runs[0].bold = True; p.runs[0].font.size = Pt(17); p.runs[0].font.color.rgb = NAVY
p = P("Project Report", "c", after=2); p.runs[0].bold = True; p.runs[0].font.size = Pt(12)
P("Vihaan Gandhi & Jash Visaria", "c", after=2, size=11.5)
P("Roll Nos: E002 & E003", "c", after=2, size=10.5)
P("Date: 20th October, 2026", "c", after=6, size=10.5)
rule = doc.add_paragraph(); pPr = rule._p.get_or_add_pPr(); pb = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
for k, v in (("val", "single"), ("sz", "12"), ("space", "1"), ("color", "1F3864")): bt.set(qn(f"w:{k}"), v)
pb.append(bt); pPr.append(pb); rule.paragraph_format.space_after = Pt(4)

# ================================================================== ABSTRACT
doc.add_heading("Abstract", 1)
P(f"Output from solar plants is variable, strongly periodic and contains many exact zeros, so standard statistical tools cannot be applied to it directly. "
  f"This study tests which probability models and time-series models describe PV output best, using maximum likelihood estimation (MLE) and information criteria (AIC and BIC). "
  f"The data are the 10-minute output of {K['data_checks']['n_plants']} Alabama PV plants for 2006 (NREL Solar Power Data for Integration Studies, as redistributed in the LSTNet repository), "
  f"aggregated to {PRE['n_hours']:,} hourly values. Because {100*PRE['share_hours_exact_zero']:.1f}% of hours are night-time zeros, the analysis works on a clear-sky index (output divided by an empirical "
  f"clear-sky envelope) on daylight hours (n = {PRE['n_in_sample']:,}). ADF and KPSS tests show that raw output is not stationary, while the index is stationary once an annual harmonic is removed. "
  f"MLE was implemented from explicit log-likelihoods for seven distributions and validated against SciPy, and ARMA and GARCH models were fitted by exact maximum likelihood. "
  f"The data are NREL-modelled (simulated) output, and the timestamps are reconstructed from the file layout.")
P(f"**A two-component normal mixture is the best distribution by both AIC and BIC** (log-likelihood {mcs.loc['2-component normal mixture','loglik']:.0f}; ΔBIC to the runner-up = {runner_up:,.0f}), "
  f"and the best single family is the scaled Beta. This ranking holds in all eight sensitivity checks. All bootstrap goodness-of-fit tests nevertheless reject every model, which is expected with "
  f"strong serial dependence. A seasonal ARMA model ARMA(2,1)×(1,0,0)₂₄ captures most of the dependence (log-likelihood {TSK['best_loglik']:.0f} versus about 1,001 for the best i.i.d. model) and "
  f"cuts one-hour-ahead RMSE by {skill:.1f}% relative to persistence of the index. However, all residual diagnostics for ARMA and GARCH models reject, and AIC and BIC disagree on the annual cycle. "
  f"The conclusion is that the mixture and ARMA models are useful descriptions and predictors, but none of them is a correct generating model.")

# ================================================================== 1 INTRODUCTION
doc.add_heading("1. Introduction", 1)
P("Solar generation depends on the sun, the season and the weather. Planning reserves, storage and grid operation needs a quantitative description of this variability. "
  "Two descriptions are useful. A marginal distribution says how likely each output level is, regardless of when it occurs. A dynamic model says how the output at one hour depends on the preceding hours. "
  "Maximum likelihood estimation is the standard method for fitting both, and AIC and BIC give a principled way to choose between competing models.")
P("Renewable-energy data do not automatically satisfy the assumptions behind these tools. Night-time output is exactly zero, the series has deterministic daily and annual cycles, and neighbouring hours are "
  "highly correlated. A large part of this project therefore consists of testing these assumptions, transforming the data until the methods apply, and reporting honestly where they still do not.")
P("The study covers the following steps:")
NLIST(["**Stationarity testing:** ADF and KPSS tests, rolling statistics and group tests (hour of day, month) on raw output and on the clear-sky index.",
       "**Maximum likelihood estimation:** Normal, Student-t, Log-normal, Gamma, Weibull, scaled Beta and a two-component normal mixture, with convergence checks and validation.",
       "**Model selection:** AIC, BIC, Akaike weights and parametric-bootstrap Kolmogorov-Smirnov and Anderson-Darling tests that re-estimate the parameters.",
       "**Time-series models:** ARMA and seasonal ARMA by exact Gaussian likelihood, GARCH(1,1) on the innovations, and Ljung-Box, ARCH-LM and Jarque-Bera diagnostics.",
       "**Forecasting:** a chronological one-hour-ahead test with order selection on training data only."])
P("The research questions are: (i) Is raw PV output stationary, and if not, what restores stationarity? (ii) Which distribution is best according to AIC and BIC, and is the ranking robust? "
  "(iii) Does any candidate pass a goodness-of-fit test when its parameters are estimated? (iv) Do ARMA and GARCH models pass residual diagnostics? (v) Does the selected dynamic model improve forecasts over simple baselines?")

# ================================================================== 2 METHODOLOGY
doc.add_heading("2. Methodology", 1)
doc.add_heading("2.1 Data Collection", 2)
dc = K["data_checks"]
P("The data are the **Alabama solar power data for 2006** from the U.S. National Renewable Energy Laboratory (NREL) *Solar Power Data for Integration Studies*. "
  "The copy used is the \"solar-energy\" file of the public LSTNet repository (Lai et al., 2018), downloaded on 2026-10-19 "
  "(solar_AL.txt.gz; 8,748,993 bytes; SHA-256 1ae9329c…20489a1c). Each column is one plant and each row one 10-minute step. "
  "The analysis variable is the fleet total (sum over the 137 plants) averaged to hourly values.".replace("*", ""))
TABLE(pd.DataFrame([
    ("Source", "NREL Solar Power Data for Integration Studies (Alabama, 2006), via the LSTNet repository"),
    ("Size / frequency", f"{dc['n_rows']:,} rows × {dc['n_plants']} plants; 10-minute steps (365 × 144 rows)"),
    ("Period", f"{dc['first_timestamp'][:10]} to {dc['last_timestamp'][:10]} (reconstructed time axis)"),
    ("Quality checks", f"{dc['n_missing']} missing, {dc['n_negative']} negative, {dc['n_duplicate_timestamps']} duplicates, {dc['n_plants_always_zero']} always-zero plants"),
    ("Units / variable", "MW; fleet total, averaged to 8,760 hourly values"),
    ("Analysis series", f"Clear-sky index on daylight hours, n = {PRE['n_in_sample']:,}"),
    ("Train / test split (forecast)", f"Train: to {FC['train_end'][:10]} ({FC['train_days']} days); test: {FC['test_days']} days ({FC['n_test']} daylight hours)")],
    columns=["Item", "Detail"]), "Dataset description", widths=[4.2, 11.8], size=9, left_cols=(0, 1))
P("**Two limitations of the data are important.** First, to the best of our knowledge the NREL integration-study data are *modelled* PV output derived from satellite solar-radiation and weather data for hypothetical plants, "
  "not metered generation; the NREL page could not be opened from our environment, so this should be confirmed at https://www.nrel.gov/grid/solar-power-data.html. "
  f"Second, the file has no timestamps. We assumed that row 0 is 2006-01-01 00:00 and that rows follow a regular 10-minute grid. The data support this: the mean daily profile peaks at {K['time_axis_evidence']['peak_time']}, and the number of hours "
  f"with positive output per day is {K['time_axis_evidence']['daylength_dec']:.1f} in December and {K['time_axis_evidence']['daylength_jun']:.1f} in June, which matches the astronomical day length at about 32-33°N.".replace("*", ""))

doc.add_heading("2.2 Pre-processing and Exploratory Data Analysis", 2)
d = K["eda"]["describe"]
P(f"No values were missing, negative or duplicated, so no imputation was needed. Hourly fleet output has mean {d['mean']:.0f} MW, standard deviation {d['std']:.0f} MW and maximum {d['max']:.0f} MW (Figure 1). "
  f"Its median is only {PRE['hourly_median_mw']:.1f} MW because **{PRE['n_hours_exact_zero']:,} hours ({100*PRE['share_hours_exact_zero']:.1f}%) are exactly zero**; the estimated probability of zero is "
  f"{K['zero_mass']['p0_mle']:.3f} (SE {K['zero_mass']['se']:.3f}). These zeros are structural (night), not errors. A continuous density cannot assign probability to a single point, so "
  "the raw data are kept untouched but distribution fitting and time-series modelling use the clear-sky index described below.")
FIGURE("fig01_hourly_series_overview.png", "Hourly fleet output in 2006: full year, a winter week and a summer week", 14.8)
P("**Clear-sky index.** For every hour of the year, the envelope E_t is the 95th percentile of output at the same hour of day within ±7 days (circular in the year). The index is")
EQ("CSI_t = P_t / E_t,   for hours where E_t ≥ 10% of its annual maximum", "1")
P(f"This keeps {PRE['n_daylight_hours']:,} daylight hours. {PRE['n_daylight_zero_output']} of them have exactly zero output and were dropped (several candidate densities have no mass at 0), leaving n = {PRE['n_in_sample']:,}. "
  f"The index has mean {IDX['mean']:.3f}, SD {IDX['sd']:.3f}, skewness {IDX['skewness']:.2f} and median {IDX['quantiles']['0.5']:.3f}. {PRE['n_index_above_1']} values exceed 1 (maximum {PRE['max_index']:.2f}), which is natural for a 95th-percentile envelope. "
  f"The monthly mean index ranges from {min(IDX['monthly_mean'].values()):.2f} (February) to {max(IDX['monthly_mean'].values()):.2f} (June), and its SD is about 0.29-0.30 in winter but 0.11-0.15 in summer, so weather variability is larger in the cool season. "
  "For the forecasting experiment a causal envelope (trailing 14 days) is used so that no future information enters the index.")
FIGURE("fig04_clearsky_envelope_and_index.png", "Clear-sky envelope and the resulting clear-sky index", 14.8)

doc.add_heading("2.3 Stationarity Testing", 2)
P("A series is weakly stationary if its mean and variance are constant and its autocovariance depends only on the lag. Two tests were used at the 5% level. "
  "**ADF** (Dickey and Fuller, 1979): H0 = unit root (non-stationary), H1 = no unit root; a small p-value supports stationarity. "
  "**KPSS** (Kwiatkowski et al., 1992): H0 = level-stationary, H1 = unit root; a small p-value is evidence against stationarity. "
  "If ADF rejects and KPSS does not, the series is stationary. If both reject, there is no unit root but the mean is not constant, typically because of a deterministic cycle. "
  "Hour-of-day and month ANOVA, Kruskal-Wallis and Levene tests were also run to test directly for periodic changes in mean and variance.")
a = STAT[STAT.test == "ADF"].set_index("series"); kp = STAT[(STAT.test == "KPSS") & STAT.specification.str.contains("regression='c', bandwidth=auto")].set_index("series")
short = {"Stationary (ADF rejects unit root, KPSS does not reject stationarity)": "Stationary", "Non-stationary (unit root not rejected, KPSS rejects stationarity)": "Non-stationary"}
rows = []
for s_ in VERD.series:
    v = VERD.set_index("series").loc[s_, "verdict"]
    rows.append((s_.replace("Clear-sky index", "CSI"), a.loc[s_, "statistic"], a.loc[s_, "p_value"], kp.loc[s_, "statistic"], kp.loc[s_, "p_value"], short.get(v, "No unit root, mean not constant")))
t2 = pd.DataFrame(rows, columns=["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p", "Conclusion"])
t2["ADF p"] = t2["ADF p"].map(lambda x: "< 0.001" if x < 1e-3 else f"{x:.3f}")
TABLE(t2, "Stationarity tests (CSI = clear-sky index; KPSS p-values are limited to [0.01, 0.10])", widths=[5.4, 1.8, 1.6, 1.8, 1.6, 3.8], size=8.5,
      fmt={"ADF stat": "{:.2f}", "KPSS stat": "{:.3f}", "KPSS p": "{:.3f}"})
P(f"Raw hourly output has no unit root (ADF p < 0.001) but is strongly non-stationary in the mean (KPSS statistic {kp.iloc[0].statistic:.2f}), because of the deterministic daily and annual cycles. "
  f"Hour-of-day ANOVA (F = {PER.iloc[0].statistic:.0f}) and Levene's test ({PER.iloc[2].statistic:.0f}) confirm that both the mean and the variance change within the day. Seasonal differencing at lag 24 restores stationarity, which shows that the daily cycle is the main cause. "
  f"The clear-sky index removes the daily pattern (index vs hour of day: ANOVA p = {PER.iloc[6].p_value:.2f}, Levene p = {PER.iloc[8].p_value:.2f}), but it still varies with the month (both p < 0.001). "
  f"After removing one annual harmonic (R² = {K['stationarity']['harmonic_r2']:.3f}) it passes both ADF and KPSS. Its variance still differs between months, so it is stationary in the mean but not strictly in the variance. "
  f"The autocorrelation is very high (lag 1: {ACF['index_lag1']:.3f}; lag 24: {ACF['index_lag24']:.3f}; Figure 3), but day-to-day persistence is weaker: the lag-1 autocorrelation of daily means is {K['effective_sample']['rho_daily_lag1']:.2f}, "
  f"which gives only about {K['effective_sample']['n_eff_days_ar1_approx']:.0f} effectively independent days.")
FIGURE("fig06b_acf_pacf_clearsky_index.png", "ACF and PACF of the clear-sky index (pairwise-complete lags; PACF shown to lag 6 where the estimate is reliable)", 14.2)

doc.add_heading("2.4 Models Applied", 2)
doc.add_heading("2.4.1 Maximum likelihood estimation", 3)
P("For independent observations x₁,…,xₙ from a density f(x; θ), the log-likelihood is ℓ(θ) = Σ ln f(xᵢ; θ) and the maximum likelihood estimator is θ̂ = argmax ℓ(θ). "
  "Under regularity conditions θ̂ is approximately normal with covariance equal to the inverse observed information:")
EQ("θ̂ ≈ N(θ, I(θ̂)⁻¹),   I(θ̂) = −∂²ℓ/∂θ∂θᵀ evaluated at θ̂", "2")
P("Standard errors are the square roots of its diagonal and 95% intervals are θ̂ ± 1.96·SE. Each log-likelihood was coded explicitly, parameters were mapped to unconstrained coordinates (log for positive parameters, logit for weights), "
  "and ℓ was maximised numerically with BFGS. A fit was accepted as converged only if the optimiser reported success, the gradient norm was small and the numerical Hessian of −ℓ was positive definite. "
  "Closed-form results (Normal, log-normal and the Gamma profile equation) were checked, and all fits were compared with scipy.stats.")
doc.add_heading("2.4.2 Candidate distributions", 3)
TABLE(pd.DataFrame([
    ("Normal", "(2πσ²)^(-1/2) exp(−(x−μ)²/2σ²)", "μ, σ", "2"),
    ("Student-t", "Γ((ν+1)/2) / (Γ(ν/2)√(νπ)σ) · (1+z²/ν)^(−(ν+1)/2)", "μ, σ, ν", "3"),
    ("Log-normal", "1/(xσ√2π) exp(−(ln x − μ)²/2σ²)", "μ, σ", "2"),
    ("Gamma", "x^(a−1) e^(−x/s) / (Γ(a) s^a)", "shape a, scale s", "2"),
    ("Weibull", "(c/λ)(x/λ)^(c−1) exp(−(x/λ)^c)", "shape c, scale λ", "2"),
    ("Scaled Beta on (0, c)", "(x/c)^(α−1)(1−x/c)^(β−1) / (c·B(α,β))", "α, β (c = 1.05·max)", "2"),
    ("Normal mixture (2 comp.)", "w·N(μ₁,σ₁²) + (1−w)·N(μ₂,σ₂²)", "w, μ₁, σ₁, μ₂, σ₂", "5")],
    columns=["Model", "Density", "Parameters", "k"]), "Candidate distributions fitted to the clear-sky index", widths=[3.8, 7.2, 3.6, 1.0], size=8.5, left_cols=(0, 1))
P("The Beta is fitted to x/c with the Jacobian term −n ln c, so that its log-likelihood is comparable with the others; c is fixed and not estimated. "
  "The mixture likelihood is not concave, so it was initialised by the EM algorithm (Dempster et al., 1977) from several starting points and then polished with BFGS, with a lower bound of 10⁻⁴ on each σ. "
  "The mixture represents two weather states, a broad \"variable sky\" component and a narrow \"clear sky\" component; these labels are an interpretation of the fitted components.")
doc.add_heading("2.4.3 ARMA and GARCH models", 3)
P("The clear-sky index is modelled as a regression with seasonal ARMA errors on the hourly grid, with night hours treated as missing values:")
EQ("φ(B) Φ(B²⁴) (Y_t − μ_t) = θ(B) ε_t,   ε_t ~ N(0, σ²),   μ_t = c (+ a·cos(2πd/365) + b·sin(2πd/365))", "3")
P("The exact Gaussian likelihood is computed by the Kalman filter, which handles the missing night values. It uses the same n = 3,846 observations as the i.i.d. fits, so that log-likelihoods, AIC and BIC are directly comparable. "
  "GARCH(1,1) (Engle, 1982; Bollerslev, 1986) lets the innovation variance evolve, σ_t² = ω + α ε²_{t−1} + β σ²_{t−1}, and was fitted to the ARMA innovations with normal and Student-t errors.")

doc.add_heading("2.5 Model Selection and Evaluation", 2)
EQ("AIC = −2ℓ(θ̂) + 2k,    BIC = −2ℓ(θ̂) + k ln n", "4")
P("AIC (Akaike, 1974) targets predictive accuracy and BIC (Schwarz, 1978) penalises complexity more heavily for large n. Differences ΔAIC and ΔBIC to the best model and Akaike weights are reported. "
  "Information criteria rank models but do not show that the best model fits, so goodness of fit was also tested. Standard Kolmogorov-Smirnov (KS) and Anderson-Darling (AD) critical values are invalid when parameters are estimated from the same data (Lilliefors, 1967), "
  "so the null distribution was obtained by a parametric bootstrap with re-estimation in each replicate (B = 200; smallest attainable p = 1/201 = 0.005). "
  "The robustness of the ranking was checked by changing the envelope quantile (0.90, 0.98), the daylight threshold (5%, 20%), the envelope window (±15 days), and by using noon values only (n = 365) and every third day, which removes most of the dependence. "
  "Residuals were checked with the Ljung-Box test (Ljung and Box, 1978) on residuals and squared residuals, the ARCH-LM test and the Jarque-Bera test. "
  f"Only the forecasting experiment uses held-out data: a chronological {int(100*K['settings']['train_fraction'])}/{100-int(100*K['settings']['train_fraction'])} train/test split of the days (Section 3.5), while the distribution and ARMA/GARCH comparisons use the whole year. The model order is chosen by BIC on the training data only, parameters are then applied with the filter so that each forecast uses data up to the previous hour, "
  "and forecasts are scored by RMSE and MAE in index units and MW against persistence baselines. "
  "All code is in the project repository (https://github.com/vihaang123/Maximum-likelihood-estimation-renewable-energy); the random seed is fixed in src/config.py. "
  f"The software used was Python {ENV['python']}, NumPy {ENV['numpy']}, pandas {ENV['pandas']}, SciPy {ENV['scipy']}, statsmodels {ENV['statsmodels']} and arch {ENV['arch']}.")

# ================================================================== 3 RESULTS
doc.add_heading("3. Results", 1)
doc.add_heading("3.1 Maximum Likelihood Estimates", 2)
P(f"All seven models converged (positive-definite Hessian) and the log-likelihoods agree with SciPy to within 10⁻⁴ or better (largest parameter difference {VALID.max_abs_param_difference.max():.1e}). "
  "For the mixture, SciPy has no fitter, so only its log-likelihood value was recomputed independently. Table 4 lists the estimates. The standard errors assume independent observations and are therefore too small (Section 2.3).")
e = MLE.copy(); e["Estimate (SE)"] = [f"{x:.4f} ({s:.4f})" for x, s in zip(e.estimate, e.std_error)]; e["95% CI"] = [f"[{l:.4f}, {h:.4f}]" for l, h in zip(e.ci95_low, e.ci95_high)]
TABLE(e[["model", "parameter", "Estimate (SE)", "95% CI"]].rename(columns={"model": "Model", "parameter": "Parameter"}), "Maximum likelihood estimates (clear-sky index, n = 3,846)",
      widths=[4.6, 2.6, 4.0, 4.6], size=8.5, left_cols=(0, 1))
P(f"The mixture puts weight {mix['weight1']:.3f} on a broad component (mean {mix['mu1']:.3f}, SD {mix['sigma1']:.3f}) and weight {1-mix['weight1']:.3f} on a narrow component close to the envelope (mean {mix['mu2']:.3f}, SD {mix['sigma2']:.3f}). "
  "Figure 4 shows that the data are clearly bimodal, which no unimodal family can reproduce.")
FIGURE("fig08_fitted_densities.png", "Fitted densities over the histogram of the clear-sky index", 14.8)

doc.add_heading("3.2 Model Selection and Goodness of Fit", 2)
m = MC[["model", "k", "loglik", "AIC", "BIC", "delta_BIC", "KS_statistic", "KS_p_bootstrap"]].copy()
m["KS_p_bootstrap"] = m["KS_p_bootstrap"].map(lambda x: f"{x:.3f}")
TABLE(m.rename(columns={"model": "Model", "loglik": "Log-lik", "delta_BIC": "ΔBIC", "KS_statistic": "KS D", "KS_p_bootstrap": "Boot. p"}),
      "Distribution comparison and bootstrap goodness of fit (n = 3,846)", widths=[4.4, 0.8, 2.0, 2.0, 2.0, 1.9, 1.5, 1.6], size=8.5, bold_rows=(0,),
      fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔBIC": "{:.1f}", "KS D": "{:.3f}"})
P(f"The mixture is best by AIC and BIC, with ΔBIC = {runner_up:,.0f} to the scaled Beta, which is the best single family. The order of the single families is Beta, Student-t, Normal, Weibull, Gamma, Log-normal; "
  "the log-normal and Gamma are poor because they are right-skewed whereas the index is left-skewed (skewness −1.12). "
  "**Every model is rejected by the bootstrap KS and AD tests** (p = 0.005, the smallest value attainable with B = 200). The KS distance of the mixture (0.036) is far smaller than for the others (0.14-0.21), so it is a much better description, but not an exact one. "
  "With n = 3,846 the tests detect very small departures, and the serial dependence means that the i.i.d. bootstrap null does not reproduce the data, so no model is claimed to fit adequately.")
s = SENS[["variant", "n", "best_by_BIC", "best_single_family_BIC", "delta_BIC_runner_up"]].copy()
s["best_by_BIC"] = s["best_by_BIC"].str.replace("2-component normal mixture", "Mixture")
TABLE(s.rename(columns={"variant": "Variant", "best_by_BIC": "Best by AIC and BIC", "best_single_family_BIC": "Best single family", "delta_BIC_runner_up": "ΔBIC to runner-up"}),
      "Sensitivity of the ranking to pre-processing choices", widths=[6.0, 1.2, 3.0, 3.2, 2.6], size=8.5, fmt={"ΔBIC to runner-up": "{:.0f}"}, left_cols=(0,))
P("The mixture ranks first by both criteria in all eight variants, including noon-only values and every third day, which remove most of the serial dependence. "
  "The best single family is Beta in seven variants and Student-t when the envelope quantile is lowered to 0.90.")

doc.add_heading("3.3 Time-Series Models", 2)
tsc = TS[TS.model.isin(["iid Gaussian + annual cycle", "AR(1)", "AR(2)", "ARMA(1,1)", "ARMA(2,1)", "ARMA(1,1)x(1,0,0)24", "ARMA(2,1)x(1,0,0)24", "ARMA(2,1)x(1,0,0)24, no annual cycle"])][["model", "k", "loglik", "AIC", "BIC", "delta_BIC"]]
tsc = tsc.sort_values("BIC").reset_index(drop=True)
TABLE(tsc.rename(columns={"model": "Model", "loglik": "Log-lik", "delta_BIC": "ΔBIC"}), "ARMA-family models (n = 3,846; same observations as the i.i.d. fits)",
      widths=[6.0, 0.9, 2.2, 2.4, 2.4, 2.0], size=8.5, bold_rows=(0,), fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔBIC": "{:.1f}"})
P(f"Dependence dominates: the best ARMA log-likelihood ({TSK['best_loglik']:.0f}) is about six times the best i.i.d. value (1,001). The seasonal term at lag 24 is the most useful addition (ΔBIC of about 590 for ARMA(1,1) with it versus without it). "
  f"**BIC and AIC disagree**: BIC selects the model without an annual cycle (ΔBIC = 6.0 in its favour), AIC selects the one with it (ΔAIC = 6.5). The likelihood-ratio test of the annual harmonic gives LR = {TSK['lr_test_annual_cycle']['statistic']:.2f} on 2 df, "
  f"p = {TSK['lr_test_annual_cycle']['p_value']:.4f}, but this test treats the data as independent given the model, so the evidence for an annual cycle in the mean is modest.")
tp = TSP.copy()
tp["parameter"] = tp["parameter"].map({"const": "constant", "ar.L1": "AR(1) φ₁", "ar.L2": "AR(2) φ₂", "ma.L1": "MA(1) θ₁", "ar.S.L24": "Seasonal AR(24) Φ", "sigma2": "σ²"}).fillna(tp["parameter"])
TABLE(tp[["parameter", "estimate", "std_error", "z", "ci95_low", "ci95_high"]].rename(columns={"parameter": "Parameter", "estimate": "Estimate", "std_error": "SE", "ci95_low": "CI low", "ci95_high": "CI high"}),
      f"Parameters of the BIC-selected model {TSK['best_BIC']}", widths=[3.8, 2.4, 2.2, 2.2, 2.6, 2.6], size=8.5, fmt={"Estimate": "{:.4f}", "SE": "{:.4f}", "z": "{:.1f}", "CI low": "{:.4f}", "CI high": "{:.4f}"})
P(f"The estimates describe a persistent but stationary process (φ₁ + φ₂ = {TSP.set_index('parameter').loc['ar.L1','estimate']+TSP.set_index('parameter').loc['ar.L2','estimate']:.3f} < 1) with day-to-day persistence at the same hour (Φ = 0.42). "
  f"For volatility, GARCH(1,1) with Student-t errors is preferred by a wide margin (log-likelihood {GC.iloc[0].loglik:.0f} vs {GC.iloc[2].loglik:.0f} for constant variance), but the fit is degenerate: α = {gp['alpha[1]']:.3f}, β = {gp['beta[1]']:.3f} (α + β = 1) and ν = {gp['nu']:.2f} degrees of freedom. "
  "This suggests that the recursion is absorbing a different structure, namely level-dependent variance (Section 3.4), rather than true volatility clustering.")

doc.add_heading("3.4 Residual Diagnostics", 2)
def diag_rows(mask, label):
    d_ = DIAG[mask]
    out = []
    for test, lag in (("Ljung-Box", 24.0), ("Ljung-Box (squared residuals)", 24.0), ("ARCH-LM", 12.0), ("Jarque-Bera", None)):
        r_ = d_[(d_.test == test) & ((d_.lag == lag) if lag else True)].iloc[0]
        out.append((label, test + (f" (lag {int(lag)})" if lag else ""), r_.statistic, pf(r_.p_value), "Yes" if r_.reject_H0 else "No"))
    return out
is_g = DIAG.model.str.contains("GARCH", regex=False)
dd = pd.DataFrame(diag_rows(DIAG.model.str.contains("no annual cycle", regex=False) & ~is_g, "ARMA(2,1)×(1,0,0)₂₄") + diag_rows(is_g, "GARCH(1,1)-t (standardised)"),
                  columns=["Model", "Test", "Statistic", "p-value", "Reject H0"])
TABLE(dd, "Residual diagnostics (H0: no autocorrelation / no ARCH effect / normality)", widths=[4.8, 4.8, 2.4, 2.2, 1.8], size=8.5, fmt={"Statistic": "{:.1f}"}, left_cols=(0, 1))
P("All 10 diagnostic tests reject for every ARMA model and for the GARCH model (full results in results/diagnostic_results.csv). **None of the fitted dynamic models is adequate in the strict sense.** Three specific features explain this:")
LIST([f"**The night gap.** The residual ACF is {RS['acf_lag1']:.2f} at lag 1 but {RS['acf_lag13']:.2f} at lag 13. On the hourly grid this links an hour to the one 13 hours earlier, which for early-morning hours lies on the previous afternoon, across the night. "
      "The ARMA structure on the 24-hour grid does not represent this dependence. Changing the assumed gap length (0, 4, 12 or 24 night slots) gave log-likelihoods of 4,907, 6,117, 6,222 and 6,223, so the true 12-slot gap is already near the optimum.",
      f"**Level-dependent variance.** The residual SD falls from {RS['std_lowest_level_quintile']:.2f} in the lowest quintile of fitted level to {RS['std_highest_level_quintile']:.2f} in the highest: variability is large when the index is expected to be low and small when it is near 1. "
      "A logit transformation to stabilise the variance was tried, but it gave a lower log-likelihood on the original scale (6,012 vs 6,222) and was rejected.",
      "**Non-normal errors.** Residual skewness is −0.58 and excess kurtosis 2.04, so Jarque-Bera rejects."], numbered=False)
FIGURE("fig12_arma_residual_diagnostics.png", "Residual diagnostics of the selected ARMA model", 14.8)

doc.add_heading("3.5 Out-of-Sample Forecasting", 2)
SPL = rd("train_test_split.csv")
P("**Where the train/test split is used.** A chronological split was used for the forecasting experiment only. The first 75% of days form the training set and the last 25% the hold-out test set; the data are never shuffled, and no test data are used for estimation or order selection. "
  "The distribution fits (Sections 3.1 and 3.2) and the ARMA and GARCH model comparisons (Section 3.3) describe the whole year, so they are in-sample comparisons by AIC and BIC; they are not scored on held-out data.")
spl = SPL.copy(); spl["start"] = spl["start"].astype(str); spl["end"] = spl["end"].astype(str)
spl["used_for"] = spl["used_for"].str.replace("one-hour-ahead scoring", "forecast scoring", regex=False)
TABLE(spl.rename(columns={"set": "Set", "start": "Start", "end": "End", "days": "Days", "daylight_observations": "Daylight hours", "used_for": "Used for"}),
      "Chronological train/test split used for forecasting", widths=[2.6, 2.1, 2.1, 1.2, 1.9, 6.1], size=8.5, left_cols=(0, 5))
P(f"Training covers the first {FC['train_days']} days (to {FC['train_end'][:10]}) and testing the last {FC['test_days']} days (from {FC['test_start'][:10]}), so the test period is October to December. "
  f"The table counts daylight hours with a valid index; the fitted training sample contained {FC['train_nobs']:,} observations, and {FC['n_test']} test hours were scored (those whose previous hour is also observed). BIC on the training data selected ARMA(1,1)×(1,0,0)₂₄ without harmonics (Table 11).")
FIGURE("fig15_train_test_split.png", "Chronological train/test split of the clear-sky index (causal-envelope version used for forecasting)", 14.8)
fs = FS.rename(columns={"method": "Method", "RMSE_MW": "RMSE (MW)", "MAE_MW": "MAE (MW)", "RMSE_index": "RMSE (index)", "MAE_index": "MAE (index)", "RMSE_skill_vs_index_persistence": "RMSE skill"})
fs["Method"] = fs["Method"].str.replace("Selected ARMA (ARMA(1,1)x(1,0,0)24)", "ARMA(1,1)×(1,0,0)₂₄", regex=False)
TABLE(fs[["Method", "MAE (index)", "RMSE (index)", "MAE (MW)", "RMSE (MW)", "RMSE skill"]], "One-hour-ahead forecast accuracy on the hold-out period (skill = 1 − RMSE / RMSE of index persistence)",
      widths=[4.6, 2.2, 2.3, 2.1, 2.1, 2.1], size=8.5, bold_rows=(0,), fmt={"MAE (index)": "{:.4f}", "RMSE (index)": "{:.4f}", "MAE (MW)": "{:.1f}", "RMSE (MW)": "{:.1f}", "RMSE skill": "{:.3f}"})
P(f"The ARMA model has an RMSE of {FS.iloc[0]['RMSE_MW']:.1f} MW against {FS.iloc[1]['RMSE_MW']:.1f} MW for persistence of the index (a gain of {skill:.1f}%). Both are much better than persistence of raw power ({FS.iloc[3]['RMSE_MW']:.0f} MW), which cannot anticipate sunrise or sunset. "
  "The test covers a single 91-day period (October to December) and no significance test was run, so the margin over persistence is indicative only.")
FIGURE("fig14_forecast_evaluation.png", "Forecast evaluation on the hold-out period", 14.8)

doc.add_heading("3.6 Summary of Results", 2)
TABLE(pd.DataFrame([
    ("Stationarity", "Raw output: not stationary (daily and annual cycles). Clear-sky index minus annual harmonic: stationary (ADF and KPSS)."),
    ("Best distribution", f"Two-component normal mixture (BIC {mcs.loc['2-component normal mixture','BIC']:.0f}); best single family: scaled Beta (BIC {mcs.loc['Beta (scaled)','BIC']:.0f}). Robust in 8 of 8 variants."),
    ("Goodness of fit", "All 7 models rejected by bootstrap KS and AD (p = 0.005); mixture much closer (D = 0.036)."),
    ("Best time-series model", f"ARMA(2,1)×(1,0,0)₂₄: BIC selects no annual cycle, AIC selects it (LR p = {TSK['lr_test_annual_cycle']['p_value']:.4f})."),
    ("Diagnostics", "All ARMA and GARCH residual tests reject; GARCH fit is degenerate (α + β = 1)."),
    ("Train/test split", f"Chronological 75/25: {FC['train_days']} training days, {FC['test_days']} test days (forecasting only; other comparisons are in-sample)."),
    ("Forecasting", f"RMSE {FS.iloc[0]['RMSE_MW']:.0f} MW vs {FS.iloc[1]['RMSE_MW']:.0f} MW for index persistence ({skill:.1f}% lower).")],
    columns=["Question", "Result"]), "Overall summary", widths=[3.8, 12.2], size=9, left_cols=(0, 1))

# ================================================================== 4 DISCUSSION
doc.add_heading("4. Discussion", 1)
P("The main message is that the shape of the PV output distribution is better explained by two weather states than by any single family. On clear days the index sits close to the envelope with little spread, whereas on variable or overcast days it is spread over a wide range. "
  "A unimodal density has to compromise between the two, which is why the single families, including the Beta, fall far behind. The ranking stayed the same even when most of the serial dependence was removed (noon-only values and every third day), "
  "which supports the conclusion despite the dependence.")
P("The comparison between i.i.d. and ARMA likelihoods should be read carefully. The two kinds of model answer different questions: the mixture describes the marginal distribution, the ARMA model the conditional distribution given the past. "
  "The gap in log-likelihood shows how much information the past contains, not that ARMA \"beats\" the mixture as a marginal model. A natural combination, which was not attempted, is a time-series model with Beta or mixture innovations.")
P("Model selection also needs caution. AIC and BIC for the i.i.d. models assume independent observations. With only about 165 effectively independent days, the BIC penalty based on n = 3,846 is too light and the likelihoods overstate the information in the data. "
  "That the bootstrap tests reject every model, and every ARMA and GARCH diagnostic rejects, should be read in the same light: the models are useful approximations. The diagnostics point to specific improvements, namely dependence across the night and variance that depends on the level of the index.")
P("**Limitations.** The data are modelled rather than metered output and the timestamps are reconstructed; only one year, one state and the fleet total were studied; the clear-sky envelope is empirical rather than physical, so the index depends on the envelope quantile and window; "
  "and the forecast evaluation uses a single origin and a 91-day test period.")

# ================================================================== 5 CONCLUSION
doc.add_heading("5. Conclusion", 1)
P("This project implemented maximum likelihood estimation and model selection for solar PV output from first principles and tested it on real, public data. Five conclusions follow:")
NLIST(["Raw PV output is not stationary because of the daily and annual cycles and contains 46.5% structural zeros. A clear-sky index on daylight hours, minus an annual harmonic, is stationary and is the appropriate object for fitting.",
       "A two-component normal mixture is the best distribution by AIC and BIC in the baseline and in all eight sensitivity checks; the scaled Beta is the best single family.",
       "Every distribution is rejected by parametric-bootstrap goodness-of-fit tests, mainly because of serial dependence, so the best model is only the best of those compared.",
       f"A seasonal ARMA model captures most of the dependence and reduces forecast RMSE by {skill:.1f}% relative to persistence, but it fails all residual diagnostics, as does GARCH. AIC and BIC disagree on the annual cycle.",
       "The mixture and ARMA models are useful descriptions and predictors, but none of them should be regarded as the true generating model."])

# ================================================================== 6 FUTURE SCOPE
doc.add_heading("6. Future Scope", 1)
P("Future work could (i) replace the empirical envelope with a physical clear-sky model (for example from pvlib) and use metered data when available; (ii) use a block bootstrap or models with dependent errors for valid inference and goodness-of-fit tests; "
  "(iii) build a two-time-scale model (a day-level random effect plus within-day dynamics) and allow level-dependent variance, or use Beta or mixture innovations; (iv) try regime-switching models and mixtures with more components; "
  "(v) evaluate forecasts with rolling origins and probabilistic scores such as CRPS; and (vi) apply the same pipeline to wind data, where a Weibull-based analysis is the natural starting point.")

# ================================================================== REFERENCES
doc.add_heading("References", 1)
REFS = [
 "Akaike, H. (1974). A new look at the statistical model identification. IEEE Transactions on Automatic Control, 19(6), 716–723.",
 "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307–327.",
 "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
 "Burnham, K. P., & Anderson, D. R. (2002). Model Selection and Multimodel Inference (2nd ed.). Springer.",
 "Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society, Series B, 39(1), 1–38.",
 "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association, 74(366), 427–431.",
 "Efron, B. (1979). Bootstrap methods: another look at the jackknife. The Annals of Statistics, 7(1), 1–26.",
 "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987–1007.",
 "Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. Philosophical Transactions of the Royal Society A, 222, 309–368.",
 "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.",
 "Inman, R. H., Pedro, H. T. C., & Coimbra, C. F. M. (2013). Solar forecasting methods for renewable energy integration. Progress in Energy and Combustion Science, 39(6), 535–576.",
 "Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review, 55(2), 163–172.",
 "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54(1–3), 159–178.",
 "Lai, G., Chang, W.-C., Yang, Y., & Liu, H. (2018). Modeling long- and short-term temporal patterns with deep neural networks. Proceedings of the 41st International ACM SIGIR Conference, 95–104. Data: https://github.com/laiguokun/multivariate-time-series-data",
 "Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. Journal of the American Statistical Association, 62(318), 399–402.",
 "Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297–303.",
 "National Renewable Energy Laboratory. Solar Power Data for Integration Studies. https://www.nrel.gov/grid/solar-power-data.html",
 "Schwarz, G. (1978). Estimating the dimension of a model. The Annals of Statistics, 6(2), 461–464.",
 "Seabold, S., & Perktold, J. (2010). statsmodels: econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference, 92–96.",
 "Sheppard, K. arch: ARCH models in Python (software). https://github.com/bashtage/arch",
 "Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272.",
]
for r_ in REFS:
    p = doc.add_paragraph(); runs(p, r_, size=9)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent, p.paragraph_format.space_after = Cm(0.8), Cm(-0.8), Pt(3)
    p.paragraph_format.line_spacing = 1.0

# ================================================================== APPENDIX (placed after the references, below)
def appendix():
    doc.add_heading("Appendix: Additional Outputs", 1)
    P("This appendix collects every remaining figure and table produced by the project (all files are also in results/ and results/figures/ in the repository), so that the report contains the complete set of outputs.")
    doc.add_heading("A. Additional figures", 2)
    FIGURE("fig02_diurnal_and_seasonal_pattern.png", "Diurnal profile and seasonal pattern of raw hourly output", 14.5)
    FIGURE("fig03_distribution_of_raw_output.png", "Distribution of raw hourly output (all hours and strictly positive hours)", 14.5)
    FIGURE("fig05_rolling_statistics_and_group_effects.png", "Rolling statistics and group effects (hour of day, month)", 14.5)
    FIGURE("fig06a_acf_pacf_raw_output.png", "ACF and PACF of the raw hourly output", 14.5)
    FIGURE("fig07_likelihood_surface_weibull.png", "Log-likelihood surface of the Weibull model with the MLE marked", 12.5)
    FIGURE("fig09_qq_plots.png", "Q-Q plots of the clear-sky index against each fitted distribution", 14.5)
    FIGURE("fig10_distribution_model_selection.png", "AIC and BIC comparison of the candidate distributions", 14.5)
    FIGURE("fig11_iid_vs_dynamic_models.png", "I.i.d. distribution models versus dynamic models on the same observations", 14.5)
    FIGURE("fig13_garch_conditional_volatility.png", "GARCH conditional volatility of the ARMA innovations", 14.5)
    doc.add_heading("B. Additional tables", 2)
    t = STAT.copy(); t["series"] = t["series"].str.replace("Clear-sky index", "CSI")
    t["p_value"] = t["p_value"].map(lambda x: "< 0.001" if x < 1e-3 else f"{x:.3f}")
    t["Test"] = t["test"] + " (" + t["specification"].str.replace("regression=", "").str.replace("bandwidth=", "bw=").str.replace("lags by AIC", "AIC lags") + ")"
    TABLE(t[["series", "Test", "statistic", "p_value", "crit_5pct", "reject_H0"]].rename(columns={"series": "Series", "statistic": "Statistic", "p_value": "p-value", "crit_5pct": "5% crit.", "reject_H0": "Reject H0"}),
          "Complete ADF and KPSS results (all specifications)", widths=[4.6, 5.6, 1.7, 1.7, 1.5, 1.4], size=7.5, fmt={"Statistic": "{:.3f}", "5% crit.": "{:.3f}"}, left_cols=(0, 1))
    TABLE(PER[["series", "factor", "test", "statistic", "p_value", "reject_H0"]].rename(columns={"series": "Series", "factor": "Factor", "test": "Test", "statistic": "Statistic", "p_value": "p-value", "reject_H0": "Reject H0"}).assign(**{"p-value": lambda x: x["p-value"].map(pf)}),
          "Group tests for periodic changes in mean and variance", widths=[3.4, 2.4, 3.8, 2.2, 2.2, 2.0], size=8, fmt={"Statistic": "{:.2f}"}, left_cols=(0, 1, 2))
    TABLE(VALID[["model", "our_loglik", "scipy_loglik", "loglik_difference_ours_minus_scipy", "max_abs_param_difference", "validated"]].rename(columns={"model": "Model", "our_loglik": "Own log-lik", "scipy_loglik": "SciPy log-lik", "loglik_difference_ours_minus_scipy": "Difference", "max_abs_param_difference": "Max |Δθ|", "validated": "Validated"}),
          "Validation of the own MLE code against SciPy (for the mixture only the likelihood value is recomputed independently)", widths=[4.4, 2.4, 2.4, 2.0, 2.2, 1.8], size=8, fmt={"Own log-lik": "{:.3f}", "SciPy log-lik": "{:.3f}", "Difference": "{:.1e}", "Max |Δθ|": "{:.1e}"})
    TABLE(TS[["model", "k", "loglik", "AIC", "BIC", "delta_AIC", "delta_BIC"]].rename(columns={"model": "Model", "loglik": "Log-lik", "delta_AIC": "ΔAIC", "delta_BIC": "ΔBIC"}),
          "All ARMA-family models (n = 3,846)", widths=[5.8, 0.9, 2.1, 2.3, 2.3, 1.8, 1.8], size=8, fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔAIC": "{:.1f}", "ΔBIC": "{:.1f}"})
    TABLE(GC[["model", "k", "loglik", "AIC", "BIC", "delta_BIC"]].rename(columns={"model": "Model", "loglik": "Log-lik", "delta_BIC": "ΔBIC"}),
          "Volatility models for the ARMA innovations", widths=[5.8, 0.9, 2.3, 2.5, 2.5, 2.0], size=8, fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}", "ΔBIC": "{:.1f}"})
    TABLE(GP[["model", "parameter", "estimate", "std_error", "p_value"]].rename(columns={"model": "Model", "parameter": "Parameter", "estimate": "Estimate", "std_error": "SE", "p_value": "p-value"}),
          "Parameters of the selected GARCH(1,1)-t model", widths=[5.0, 2.4, 2.6, 2.4, 2.4], size=8, fmt={"Estimate": "{:.4f}", "SE": "{:.4f}", "p-value": "{:.2g}"}, left_cols=(0, 1))
    # full diagnostics as a p-value grid
    dg = DIAG[DIAG.test.isin(["Ljung-Box", "Ljung-Box (squared residuals)", "ARCH-LM", "Jarque-Bera"])].copy()
    dg["col"] = dg.apply(lambda r: {"Ljung-Box": "LB", "Ljung-Box (squared residuals)": "LB²", "ARCH-LM": "ARCH", "Jarque-Bera": "JB"}[r.test] + ("" if pd.isna(r.lag) else f"({int(r.lag)})"), axis=1)
    grid = dg.pivot_table(index="model", columns="col", values="p_value", sort=False)
    order = [c for c in ["LB(10)", "LB(24)", "LB(48)", "LB²(10)", "LB²(24)", "LB²(48)", "ARCH(5)", "ARCH(12)", "ARCH(24)", "JB"] if c in grid.columns]
    grid = grid.reindex(columns=order).map(lambda x: "–" if pd.isna(x) else ("<0.001" if x < 1e-3 else f"{x:.3f}"))
    grid.insert(0, "Model", [m_.replace(" on ARMA(2,1)x(1,0,0)24, no annual cycle innovations (standardised)", " (std. resid.)").replace(" (i.i.d.), normal scores of PIT", " (i.i.d., PIT)") for m_ in grid.index])
    TABLE(grid.reset_index(drop=True), "Residual-test p-values for every model (LB = Ljung-Box, LB² = on squared residuals, ARCH = ARCH-LM, JB = Jarque-Bera; lag in brackets). All are below 0.05 except none.".replace(" All are below 0.05 except none.", " Every test rejects at the 5% level."),
          widths=[4.4] + [1.2] * len(order), size=7, left_cols=(0,))
    TABLE(rd("residual_variance_by_fitted_level.csv").rename(columns={"fitted_value_quintile": "Fitted-level quintile", "std_standardised_residual": "Residual SD", "count": "n", "mean_fitted_index": "Mean fitted index"}),
          "Residual standard deviation by quintile of fitted level", widths=[4, 3, 2, 3.6], size=8.5, fmt={"Residual SD": "{:.3f}", "Mean fitted index": "{:.3f}"})
    TABLE(rd("residual_variance_by_hour.csv").rename(columns={"hour": "Hour of day", "std_standardised_residual": "Residual SD", "count": "n"}),
          "Residual standard deviation by hour of day", widths=[3.5, 3, 2], size=8.5, fmt={"Residual SD": "{:.3f}"})
    TABLE(NG[["night_slots", "seasonal_period", "n", "k", "loglik", "AIC", "BIC"]].rename(columns={"night_slots": "Night slots", "seasonal_period": "Seasonal period"}),
          "Sensitivity of the ARMA fit to the assumed length of the night gap", widths=[2.4, 2.8, 1.6, 1.0, 2.4, 2.6, 2.6], size=8.5, fmt={"loglik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}"})
    TABLE(FT[["model", "n", "k", "loglik", "AIC", "BIC", "converged"]].rename(columns={"loglik": "Log-lik"}), "Order selection on the training sample (forecasting)",
          widths=[4.6, 1.3, 0.9, 2.2, 2.2, 2.2, 1.9], size=8.5, fmt={"Log-lik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}"})
    TABLE(rd("supplementary_raw_positive_mw_comparison.csv").rename(columns={"model": "Model", "loglik": "Log-lik"}), f"Supplementary comparison on raw positive output in MW (n = {K['raw_positive']['n']:,})",
          widths=[5, 3, 3], size=8.5, fmt={"Log-lik": "{:.1f}", "BIC": "{:.1f}"})
    ps = rd("plant_summary.csv")
    TABLE(pd.DataFrame([("Mean of plant means (MW)", ps.mean_mw.mean()), ("Smallest plant mean (MW)", ps.mean_mw.min()), ("Largest plant mean (MW)", ps.mean_mw.max()), ("Largest single-plant output (MW)", ps.max_mw.max()),
                        ("Average share of zero 10-minute values", ps.share_zero_10min.mean())], columns=["Statistic", "Value"]),
          "Summary of the 137 plants (full table in results/plant_summary.csv)", widths=[7, 3], size=8.5, fmt={"Value": "{:.3f}"})


appendix()

doc.save(OUT)
print("wrote", OUT, "| figures:", FIGN[0], "| tables:", TABN[0])
