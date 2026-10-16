# Chart questions to try on GEP

Manual test questions for chart reading on `GEP-Jun-2026.pdf` (World Bank
Global Economic Prospects, June 2026). The values in these charts are not printed
anywhere in the report; they exist only as bar heights. The expected answers were
measured from the PDF's own drawing (`app/rag/chart_measure.py`), and each one was
checked against the rendered chart. Page numbers are PDF pages.

**Tip:** mention the chart (*chart*, *figure*, *graph*) in the question. That's what
tells the app to look at the chart page.

## 1. Risk of debt distress (Figure 3.1.D, page 110)
Stacked bars for 2015 and 2026: in debt distress, high risk, moderate risk, low risk.

| Question | Expected answer |
|---|---|
| According to the chart, how did the share of low- and middle-income countries at moderate risk of debt distress change between 2015 and 2026? | Fell from ≈51.7% to ≈40.3%, about **−11.4 percentage points** |
| In the debt distress chart, which risk category grew the most between 2015 and 2026? | **High risk**: ≈20.6% → ≈34.5% (+13.9 pp). In debt distress also rose, ≈6.0% → ≈13.5% |
| From the chart, what share of countries was at low risk of debt distress in 2015 and in 2026? | ≈22.1% → ≈12.1%, about −10 pp |
| Show the debt distress chart for 2015 and 2026 as a table. | In distress / high / moderate / low: 2015 ≈6.0 / 20.6 / 51.7 / 22.1; 2026 ≈13.5 / 34.5 / 40.3 / 12.1 |

Cross-check: the report's text says the share in or at high risk "rose from 26
percent in 2015 to about 50 percent in 2026". The measured values give 26.6% and 48.0%.

## 2. Debt-service costs (Figure 3.1.B, page 110)
Bars for 2010–2025, as a percent of government revenues.

| Question | Expected answer |
|---|---|
| What does the debt-service costs chart show for EMDEs in 2010 and in 2025? | ≈7.0% of government revenues in 2010, rising to ≈11.0% in 2025 (an estimate) |
| According to the chart, in which year were debt-service costs lowest? | **2012**, ≈6.2% of government revenues |

## 3. Sovereign spreads in stress events (Figure 3.1.C, page 110)
Grouped bars: investment grade vs non-investment grade, in two stress events.

| Question | Expected answer |
|---|---|
| According to the chart, how much did sovereign spreads widen for investment-grade and non-investment-grade EMDEs during the 2020 pandemic? | Investment grade ≈189 basis points; non-investment grade ≈523 basis points (about 2.8 times as much) |
| In the same chart, how did the 2022 Federal Reserve tightening cycle compare? | Investment grade ≈34 bp; non-investment grade ≈308 bp |

## 4. Composition of fiscal revenues (Figure 1.12.E, page 59)
Stacked bars per country group: tax, commodity nontax, other nontax.

| Question | Expected answer |
|---|---|
| According to the chart, what share of energy exporters' revenues comes from commodity nontax revenue? | ≈38% (tax ≈39%, other nontax ≈23%) |
| In the fiscal revenues chart, which country group relies most on tax revenue? | **Metal exporters**, ≈74% (agricultural exporters ≈71%, advanced economies ≈63%) |

## 5. Fiscal framework combinations (Figure 1.12.F, page 59)
Grouped bars per country group.

| Question | Expected answer |
|---|---|
| According to the chart, what percent of energy exporters have both a fiscal rule and a sovereign wealth fund? | ≈40% |
| In the fiscal framework chart, how do metal exporters compare with energy exporters? | Metal exporters: fiscal rule + SWF ≈28%, fiscal rule + fiscal council ≈12%, SWF + fiscal council ≈12%. Energy exporters: ≈40%, ≈6%, ≈9% |

## 6. Debt sustainability risks by commodity group (Figure 4.2.D, page 147)
Stacked bars for 2015, 2020 and 2026, for commodity exporters and importers.

| Question | Expected answer |
|---|---|
| According to the chart, what share of commodity importers was at high risk of debt distress in 2020? | ≈60% (in debt distress ≈10%) |
| In the debt sustainability risks chart, how did the moderate-risk share change for commodity exporters between 2015 and 2026? | ≈57.2% → ≈40.4%, about −17 pp |

## 7. Forecast revisions (Figure 1.1.A, page 27)
Stacked bars: downgraded, unchanged, upgraded.

| Question | Expected answer |
|---|---|
| From the chart, what share of economies had their 2026 growth forecast downgraded, for the world, advanced economies and EMDEs? | World ≈67%, advanced economies ≈74%, EMDEs ≈65% |

## 8. Potential growth and productivity (Figure B1.1.1, page 31)

| Question | Expected answer |
|---|---|
| According to the potential output growth chart, how has potential growth in EMDEs changed across 2000-09, 2010-19 and 2020-29? | ≈5.9% → ≈5.3% → ≈4.1% |
| What does the productivity growth chart show for each decade? | ≈1.4% (2000-09) → ≈1.1% (2010-19) → ≈0.8% (2020-29) |
| From the chart on the share of economies with potential growth below their 2000-09 average, what share of advanced economies is below it? | ≈96% of economies (≈89% of GDP). This chart is a **pasted picture**, so the answer must say it's an estimate from the chart image |

## What to check in each answer

- **Numbers** are close to the expected ones, within about 1 point (about 2 points
  for the picture chart).
- **Labels:** each value says *"measured from the chart"*, or *"estimated from the
  chart image"* for the picture chart.
- **Page numbers** are the PDF page (110, 59, 147), not the number printed on the
  page (for example 88).
- **The 📊 Chart read** line under the answer shows which page was used.
- **No stray `【2】` markers** appear.

## For comparison: ChatGPT on the first question

ChatGPT estimated moderate risk at ≈30% in 2015 and ≈20% in 2026, a fall of about
10 pp, judged by eye from the picture. Measuring the bars in the PDF's drawing gives
≈51.7% → ≈40.3%, which the report's own 26% → about 50% sentence confirms.
