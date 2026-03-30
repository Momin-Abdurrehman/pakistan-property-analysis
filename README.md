<p align="center">
  <h1 align="center">🏠 Pakistan House Price Prediction</h1>
  <p align="center">
    <em>Predicting house prices across 6 Pakistani cities using 29,000+ real listings from Zameen.com</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/R²-0.92-brightgreen" alt="R²">
    <img src="https://img.shields.io/badge/Holdout_Accuracy-71%25-blue" alt="Holdout">
    <img src="https://img.shields.io/badge/Houses-14,255-orange" alt="Houses">
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" alt="Dashboard">
  </p>
</p>

---

## What This Project Does

We scraped **29,220 property listings** from Zameen.com, cleaned and filtered them to **14,255 houses**, engineered features using Pakistan real estate domain knowledge, and built a **stacked ensemble model** that predicts house prices with **R² = 0.92**.

Tested on **2,093 completely unseen listings** — the model gets **71% of predictions within ±25%** of the actual price.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Model | Stacked Ensemble (RF + GB + XGBoost → Ridge) |
| Train/Test R² | 0.92 |
| Holdout R² (2,093 unseen) | 0.88 |
| Holdout accuracy (±25%) | 71% |
| Holdout median error | 14.7% |
| MAE | 0.70 Crore PKR |

---

## How It Works

```
Zameen.com  →  29,220 raw listings  →  14,255 clean houses  →  22 features  →  Stacked Ensemble  →  Price prediction
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

**Step 5 — Validate:** Test on 2,093 listings the model has never seen (scraped from different Zameen pages, zero URL overlap)

---

## Project Structure

```
├── notebooks/main.ipynb           # Complete analysis (72 cells)
├── scripts/scraper.py             # Zameen.com scraper
├── app.py                         # Streamlit dashboard
├── data/
│   ├── raw/zameen_raw_complete.csv      # 29,220 raw listings
│   ├── processed/houses_cleaned.csv     # 14,255 clean houses
│   └── test/zameen_holdout_test.csv     # 2,093 unseen test data
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

## Limitations

- **Listing prices, not sale prices** — asking prices may be inflated
- **No construction quality data** — new vs old house = 2-3x gap, not captured
- **Cross-city location bleed** — "DHA Phase 6" in Lahore and Karachi share one encoding
- **Premium outliers** — F-6/F-7 Islamabad (20-55 Cr) hard to predict with few examples
- **Marla = 225 sqft universally** — varies by city in government records

---

## Documentation

- **[ISSUES.md](ISSUES.md)** — Every problem we encountered and how we resolved it
- **[MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md)** — All modeling approaches tested with results

---

<p align="center">
  <em>DS 401 — Introduction to Data Science | NUST SEECS | Spring 2026</em>
</p>
