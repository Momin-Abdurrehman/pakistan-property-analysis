<p align="center">
  <h1 align="center">🏠 Pakistan Property Market Analysis & Price Prediction</h1>
  <p align="center">
    <em>End-to-end data science pipeline analyzing 24,000+ property listings across 6 Pakistani cities</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Scikit--Learn-1.6-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Data-Zameen.com-green" alt="Data Source">
  </p>
</p>

---

## 📋 Overview

This project develops a complete data science pipeline to analyze and predict property prices in Pakistan using real-world data scraped from **Zameen.com**, Pakistan's largest property portal. It covers data collection, cleaning, exploratory analysis, machine learning modeling, and interactive visualization.

**Key highlights:**
- **24,373 listings** scraped across **6 cities** and **3 property types** (Houses, Flats, Plots)
- Handles Pakistan-specific challenges: mixed units (Marla, Kanal, Sq. Yd., sqft), regional price variations, and inconsistent listing data
- **Random Forest model** achieves **R² = 0.77** for price prediction
- **Overpriced/undervalued detection** flags listings that deviate >20% from fair market value
- **Interactive Streamlit dashboard** with area-wise analysis and AI-powered price predictor

> **Course:** DS 401 – Introduction to Data Science | NUST SEECS | Spring 2026

---

## 🏗️ Project Structure

```
pakistan-property-analysis/
├── notebooks/
│   └── main.ipynb              # Complete analysis notebook (Sections 1-5)
├── scripts/
│   └── scraper.py              # Playwright scraper for Zameen.com
├── data/
│   ├── raw/
│   │   └── zameen_raw.csv      # Raw scraped data (24,373 listings)
│   └── processed/
│       └── zameen_cleaned.csv  # Cleaned dataset (18,400+ rows)
├── app.py                      # Streamlit dashboard
├── DS401 Term Project.pdf      # Project requirements
├── .gitignore
└── README.md
```

---

## 📊 Pipeline

### 1. Data Collection
- **Source:** [Zameen.com](https://www.zameen.com) — scraped using Playwright (headless Chromium)
- **Coverage:** Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad, Peshawar
- **Property types:** Houses (12,435), Plots (6,776), Flats (5,162)
- **Fields:** price, location, city, size (with unit), bedrooms, bathrooms, property type

### 2. Data Cleaning & Preprocessing
- **Price parsing:** "4.95 Crore" → `49,500,000 PKR`
- **Unit standardization:** Marla (×225), Kanal (×4,500), Sq. Yd. (×9) → sqft
- **Missing values:** Plot beds/baths filled with 0 (semantically correct); House/Flat nulls dropped
- **Outlier removal:** IQR method per property type + domain sanity filters
- **Feature engineering:** `price_per_sqft`, `log_price`, `size_category`, one-hot encoded city/type/location

### 3. Exploratory Data Analysis
- Statistical profiling with skewness/kurtosis
- Univariate (histograms, KDE, bar charts), bivariate (scatter, box plots), multivariate (pair plots)
- Correlation heatmap with interpretation
- All findings explicitly connected to modeling decisions

### 4. Modeling & Predictions
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Baseline (Mean) | ~0.93 | ~0.77 | ~0.00 |
| Linear Regression (scaled) | ~0.57 | ~0.43 | ~0.63 |
| Random Forest | ~0.45 | ~0.32 | ~0.77 |
| Gradient Boosting | ~0.45 | ~0.33 | ~0.77 |

- **62 features:** 3 numeric + 3 property type + 6 city + 50 top locations (one-hot)
- **StandardScaler** applied to numeric features for Linear Regression
- **GridSearchCV** hyperparameter tuning on Random Forest
- Diagnostics: residual plot, feature importance, actual vs predicted

### 5. Insights & Visualization
- **Overpriced/undervalued detection** using model predictions (±20% threshold)
- Polished dashboard-style summary visualizations
- 5 key findings tied to specific figures
- Limitations, assumptions, and next steps

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides an interactive way to explore the data and predictions.

**Tabs:**
| Tab | Description |
|-----|-------------|
| 📊 Market Overview | Price distribution, property type breakdown, size vs price scatter |
| 🗺️ City Comparison | Price/sqft by city, listings by city, city × type grouped bar |
| 🏘️ Area Analysis | Area-wise price breakdown within each city (top areas vs "Other") |
| 🔍 Overpriced vs Undervalued | AI-powered pricing status, best deals table |
| 🤖 Price Predictor | Enter property details → instant price estimate with market context |

### Run the dashboard:
```bash
cd pakistan-property-analysis
streamlit run app.py
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly playwright
playwright install chromium
```

### Run the notebook
```bash
cd notebooks
jupyter notebook main.ipynb
# Run all cells top-to-bottom
```

### Re-scrape data (optional)
```bash
python scripts/scraper.py
# OR set RUN_SCRAPER=True in the first code cell of main.ipynb
```

### Regenerate model artifacts
The model pickle (`data/processed/model_artifacts.pkl`) is not included in the repo due to size (80MB). It is automatically generated when you run Section 4 of the notebook. The dashboard requires this file — run the notebook first.

---

## ⚠️ Known Limitations

### Data
- **Listing prices, not sale prices** — Zameen.com shows asking prices which may be inflated
- **Single snapshot** — scraped at one point in time, no temporal trends
- **6 cities only** — smaller cities and rural areas not represented
- **Online bias** — traditional off-market properties are excluded

### Model
- **Cross-city location bleed** — shared location features (e.g., "DHA Phase 6") blend prices across cities. DHA Phase 6 in Lahore and Karachi are treated identically despite being different markets. Interaction features (`city × location`) would fix this at the cost of feature space explosion.
- **Bathrooms as proxy** — bathrooms has ~46% feature importance because it proxies property development level (0 = empty plot, 6+ = large house), not because bathrooms independently drive price.
- **Marla = 225 sqft universally** — actual Marla varies by city (225-272 sqft) in government records, but Zameen.com uses a standardized 225.

### Production
- Model needs regular retraining as market conditions change
- No external factors (interest rates, infrastructure, economic indicators)
- Micro-neighborhood effects not captured

---

## 🔮 Future Work

1. **More data sources** — Graana.com, OLX for broader coverage
2. **Geospatial features** — distances to schools, hospitals, highways
3. **Temporal analysis** — monthly scraping for trend detection
4. **Interaction features** — `city × location` to fix cross-city bleed
5. **XGBoost + Optuna** — advanced hyperparameter tuning
6. **Deployed web app** — production Streamlit/Flask app for real-time predictions

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn (RandomForest, GradientBoosting, LinearRegression, GridSearchCV) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Scraping** | Playwright |
| **Data Source** | Zameen.com |

---

<p align="center">
  <em>DS 401 — Introduction to Data Science | NUST SEECS | Spring 2026</em>
</p>
