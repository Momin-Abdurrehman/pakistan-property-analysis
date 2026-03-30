<p align="center">
  <h1 align="center">🏠 Pakistan House Price Analysis & Prediction</h1>
  <p align="center">
    <em>End-to-end data science pipeline predicting house prices across 6 Pakistani cities using 29,000+ scraped listings</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Scikit--Learn-1.6-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/XGBoost-Optuna-green" alt="XGBoost">
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/R²-0.93-brightgreen" alt="R²">
  </p>
</p>

---

## Overview

This project develops a complete data science pipeline to predict house prices in Pakistan using real-world data scraped from **Zameen.com**. It covers data collection, cleaning, exploratory analysis, feature engineering with domain knowledge, advanced ML modeling, and interactive visualization.

**Key highlights:**
- **29,220 listings** scraped across **6 cities**, filtered to **14,255 houses**
- **Derived geographic features** using Pakistan property market domain knowledge (DHA phases, CDA sector tiers, premium area flags)
- **Smoothed Bayesian target encoding** for 2,500+ locations — every neighborhood gets a price signal
- **Stacked ensemble** (RF + GB + Optuna-tuned XGBoost → Ridge) achieves **R² = 0.93**
- **Overpriced/undervalued detection** flags listings deviating >20% from fair market value
- **Interactive Streamlit dashboard** with area-wise analysis and AI-powered price predictor

> **Course:** DS 401 – Introduction to Data Science | NUST SEECS | Spring 2026

---

## Project Structure

```
pakistan-property-analysis/
├── notebooks/
│   └── main.ipynb                  # Complete analysis notebook
├── scripts/
│   └── scraper.py                  # Playwright scraper for Zameen.com
├── data/
│   ├── raw/
│   │   └── zameen_raw_complete.csv # All scraped data (29,220 listings)
│   └── processed/
│       └── houses_cleaned.csv      # Cleaned houses (14,255 rows)
├── app.py                          # Streamlit dashboard
├── ISSUES.md                       # Issues encountered & resolutions
├── MODEL_EXPERIMENTS.md            # All modeling experiments & comparisons
└── README.md
```

---

## Pipeline

### 1. Data Collection
- **Source:** Zameen.com — scraped using Playwright (headless Chromium)
- **29,220 total listings** across Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad, Peshawar
- **Targeted scraping** of 17 premium/underrepresented areas (F-6, F-7, F-8, Clifton, DHA phases)
- Fields: price, location, city, size (with unit), bedrooms, bathrooms

### 2. Data Cleaning
- Price parsing: "4.95 Crore" → numeric PKR
- Unit standardization: Marla×225, Kanal×4,500, Sq.Yd.×9 → sqft
- Filtered to **Houses only** (removed Plots/Flats — fundamentally different products)
- IQR outlier removal + domain sanity filters (min 200 sqft, min 5 Lakh)
- Result: **14,255 clean house listings**

### 3. Feature Engineering (22 features)
- **Numeric:** size_sqft, log_size, bedrooms, bathrooms
- **Geographic (derived from location names):**
  - `society_type`: DHA, Bahria, Askari, CDA_Sector, Private, Established, Other
  - `dha_phase`: phase number for DHA properties
  - `isb_sector_tier`: F=5 > E=4 > G=3 > I=2 > B,D=1
  - `is_premium_area`: binary flag for premium locations
  - `phase_number`: generic phase/block number
- **City:** one-hot (6 columns)
- **Location:** smoothed Bayesian target encoding (m=50, fit on training data only)

### 4. Modeling

| Model | R² |
|-------|-----|
| Baseline (Mean) | ~0.00 |
| Linear Regression (scaled) | ~0.74 |
| Random Forest (300 trees) | ~0.91 |
| Gradient Boosting | ~0.92 |
| XGBoost (Optuna, 200 trials) | ~0.93 |
| **Stacked Ensemble (RF+GB+XGB → Ridge)** | **~0.93** |

- Overfit check: train-test gap = 0.04, CV stable at 3.7% std/mean
- Tested on 29 real-world properties: **59% within ±25%, 90% within ±50%, median error 18.6%**

### 5. Insights
- Overpriced/undervalued property detection (±20% threshold)
- 5 key findings tied to specific figures
- Comprehensive limitations & next steps

---

## Dashboard

Streamlit app with 5 tabs:

| Tab | Description |
|-----|-------------|
| Market Overview | Price distribution, size vs price scatter |
| City Comparison | Price/sqft by city, listings by city |
| Area Analysis | Neighborhood-level price breakdown per city |
| Overpriced vs Undervalued | AI-powered pricing status, best deals table |
| Price Predictor | Enter property details → instant price estimate |

```bash
streamlit run app.py
```

---

## Getting Started

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly playwright xgboost optuna
playwright install chromium

# Run notebook
cd notebooks && jupyter notebook main.ipynb

# Run dashboard (requires model_artifacts.pkl — generated by notebook Section 4)
streamlit run app.py
```

---

## Known Limitations

- **Listing prices, not sale prices** — Zameen shows asking prices which may be inflated
- **Cross-city location bleed** — shared location names (DHA Phase 6) blend signals from different cities
- **Marla = 225 sqft universally** — actual varies by city (225-272)
- **No construction quality data** — new vs 20-year-old house = 2-3x price gap, not captured
- **Premium outliers** — F-6/F-7 Islamabad (20-55 Cr) and Model Town Lahore (28 Cr) remain hard to predict with <100 listings each

---

## Experiment Branches

| Branch | Approach | Median Error |
|--------|----------|-------------|
| `main` | All types, shared one-hot | 26.2% |
| `fix/city-location-interactions` | City-prefix one-hot | 30.0% |
| `experiment/target-encoding` | Target encoding | 31.5% |
| `houses-only` | Houses, target enc + XGBoost | 19.9% |
| `houses-improved` | **Houses, geo features + Optuna + stacking + more data** | **18.6%** |

See [MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md) for detailed comparison.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, Optuna |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Scraping | Playwright |

---

<p align="center">
  <em>DS 401 — Introduction to Data Science | NUST SEECS | Spring 2026</em>
</p>
