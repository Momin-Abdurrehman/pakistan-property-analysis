<p align="center">
  <h1 align="center">🏠 Pakistan House Price Prediction</h1>
  <p align="center">
    <em>Predicting house prices across 6 Pakistani cities using 29,000+ real listings from Zameen.com</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/R²-0.92-brightgreen" alt="R²">
    <img src="https://img.shields.io/badge/Holdout_Accuracy-70%25-blue" alt="Holdout">
    <img src="https://img.shields.io/badge/Houses-15,515-orange" alt="Houses">
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" alt="Dashboard">
  </p>
</p>

---

## What This Project Does

We scraped **29,220 property listings** from Zameen.com, cleaned and filtered them to **15,515 houses**, engineered features using Pakistan real estate domain knowledge, and built a **stacked ensemble model** that predicts house prices with **R² = 0.92**.

Tested on **1,949 completely unseen listings** — the model gets **70% of predictions within ±25%** of the actual price.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Model | Stacked Ensemble (RF + GB + XGBoost → Ridge) |
| Train/Test R² | 0.92 |
| Holdout R² (1,949 unseen) | 0.87 |
| Holdout accuracy (±25%) | 70.4% |
| Holdout median error | 15.2% |
| MAE | 0.78 Crore PKR |

---

## How It Works

```
Zameen.com  →  29,220 raw listings  →  15,515 clean houses  →  22 features  →  Stacked Ensemble  →  Price prediction
                  (scraping)              (cleaning)            (engineering)      (modeling)
```

**Step 1 — Scrape:** Playwright collects listings across 6 cities + 17 targeted premium areas

**Step 2 — Clean:** Parse prices ("4.95 Crore" → PKR), standardize 4 size units to sqft, remove outliers, deduplicate, filter to houses only

**Step 3 — Engineer features (22 total):**
- Physical: size, log_size, bedrooms, bathrooms
- Geographic: society_type (DHA/Bahria/Askari/CDA/Private/Established), DHA phase, CDA sector tier, premium area flag, phase number
- Location: smoothed Bayesian target encoding for 2,500+ neighborhoods
- City: one-hot encoded

**Step 4 — Model:** Compare 6 models (Baseline → Linear Regression → Random Forest → Gradient Boosting → XGBoost with Optuna → Stacked Ensemble)

**Step 5 — Validate:** Test on 1,949 listings the model has never seen (scraped in a separate session, zero URL overlap verified)

---

## Project Structure

```
├── notebooks/main.ipynb           # Complete analysis (78 cells)
├── scripts/scraper.py             # Zameen.com scraper
├── app.py                         # Streamlit dashboard
├── data/
│   ├── raw/zameen_raw_complete.csv      # 29,220 raw listings
│   ├── processed/houses_cleaned.csv     # 15,515 clean houses
│   └── test/zameen_holdout_test.csv     # 1,949 unseen test data
├── ISSUES.md                      # Problems we hit & how we solved them
└── MODEL_EXPERIMENTS.md           # All modeling experiments compared
```

---

## Dashboard

Interactive Streamlit app with 5 tabs:

| Tab | What it shows |
|-----|--------------|
| Market Overview | Price distributions, property type breakdown |
| City Comparison | Price/sqft rankings, city × type analysis |
| Area Analysis | Neighborhood-level breakdown within each city |
| Overpriced vs Undervalued | Properties flagged as over/under fair value |
| Price Predictor | Enter specs → get instant price estimate |

```bash
streamlit run app.py
```

---

## Quick Start

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly playwright xgboost optuna
playwright install chromium

cd notebooks && jupyter notebook main.ipynb   # Run all cells (generates model)
cd .. && streamlit run app.py                  # Launch dashboard
```

---

## What We Tried & Why

We iterated through multiple approaches. Each change was validated on real-world test data:

| Approach | Holdout Good (±25%) | Median Error | Why we moved on |
|----------|-------------------|--------------|-----------------|
| All types + label encoding | 50% | 26.2% | Bathrooms dominated at 46% importance — proxy for "built vs land" |
| Houses only + one-hot locations | 59% | 19.9% | Premium areas had 0-6 listings, model couldn't learn them |
| + Geographic features | 52% | 23.6% | Features helped premium areas but added noise elsewhere |
| + Optuna XGBoost + Stacking | 62% | 21.6% | R² jumped to 0.92 but still limited by data gaps |
| + Targeted scraping (16K) | 59% | 18.6% | F-8 error: 64% → 9.4%. Karachi still weakest |
| **+ Target encoding + holdout test** | **70.4%** | **15.2%** | **Final model — validated on 1,949 unseen listings** |

Full experiment details: **[MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md)**

---

## Limitations

- **Listing prices, not sale prices** — Zameen shows asking prices, which may be inflated above actual transaction values
- **No construction quality data** — a new house vs a 20-year-old house at the same address can differ 2-3x in price. Our data doesn't capture age, condition, or finish quality
- **Ultra-premium properties (50+ Crore)** — very few training examples at this price tier. Model predictions are less reliable for luxury properties
- **Single point in time** — scraped on one date, no temporal trends or seasonality captured
- **6 cities only** — smaller cities, towns, and rural areas are not represented
- **Online listings bias** — properties sold through traditional off-market channels are excluded
- **Marla = 225 sqft universally** — actual Marla varies by city in government records (225-272 sqft). We follow Zameen.com's standardized convention

---

## Documentation

- **[ISSUES.md](ISSUES.md)** — Every problem we encountered and how we resolved it
- **[MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md)** — All modeling approaches tested with results

---

<p align="center">
  <em>DS 401 — Introduction to Data Science | NUST SEECS | Spring 2026</em>
</p>
