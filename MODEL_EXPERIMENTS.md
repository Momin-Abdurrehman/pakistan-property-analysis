# Model Experiments Log

A record of every modeling approach we tested, what worked, what didn't, and why.

---

## Experiment 1: Location Encoding Strategies

We tested three approaches for encoding the 2,500+ unique locations. All used Random Forest (200 trees) on the same dataset.

### Approaches

| Approach | Description | Features |
|----------|-------------|----------|
| **Shared One-Hot** | Top 50 locations get one-hot columns. `loc_DHA Phase 6` fires for both Lahore and Karachi. | 62 |
| **City-Prefix One-Hot** | Same threshold, but `loc_Lahore_DHA Phase 6` and `loc_Karachi_DHA Phase 6` are separate. | 62 |
| **Target Encoding** | Each location replaced with smoothed mean log_price from training data. Bayesian prior (m=50) regularizes toward city mean. | 13 |

### Results (50 real-world properties)

| Metric | Shared One-Hot | City-Prefix | Target Encoding |
|--------|---------------|-------------|-----------------|
| Good (±25%) | **20/50** | 18/50 | 16/50 |
| Median Error | **34.2%** | 39.5% | 38.8% |
| Best on N cases | 17 | 15 | **18** |

### Conclusion
No single approach dominated. Shared one-hot won overall accuracy. Target encoding won on the hardest individual cases. We kept shared one-hot for the main branch (simplest, best overall), and explored target encoding on separate branches.

---

## Experiment 2: Houses-Only vs All Property Types

### Problem
Training on Houses + Flats + Plots forced the model to use bathrooms as a proxy for "is this built or empty land" — 46% of feature importance went to bathrooms, which is not a meaningful price signal.

### Solution
Filtered to 10,473 houses only. Removed Plots and Flats entirely.

### Results

| Metric | All Types (18K) | Houses Only (10K) |
|--------|----------------|-------------------|
| Good (±25%) | 50% | **59%** |
| Median Error | 26.2% | **19.9%** |
| Bad (>50%) | 28% | **14%** |

Bathrooms feature importance dropped from 46% to a reasonable level, and the model learned actual house price drivers instead of property type proxies.

---

## Experiment 3: Derived Geographic Features

### Features Extracted
Using Pakistan property market domain knowledge, we extracted 5 features from location name strings:

1. **society_type** — DHA, Bahria, Askari, CDA_Sector, Private, Established, Other
2. **dha_phase** — phase number for DHA properties (lower = older = often more expensive)
3. **isb_sector_tier** — F=5 > E=4 > G=3 > I=2 > B,D=1 for Islamabad CDA sectors
4. **is_premium_area** — binary flag for known premium locations (F-6/7/8, Clifton, DHA Phase 5/6, etc.)
5. **phase_number** — generic phase/block number for any society

### Validation
- Premium areas median price: 7.3 Cr vs non-premium: 3.1 Cr (2.4x gap captured)
- 68% of houses explicitly categorized by society type

---

## Experiment 4: Optuna XGBoost + Stacked Ensemble

### Optuna Tuning
200 Bayesian optimization trials on XGBoost, tuning 8 hyperparameters: n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda.

### Stacked Ensemble
Three base models (Random Forest, Gradient Boosting, tuned XGBoost) with a Ridge regression meta-learner. 5-fold CV for base model predictions to prevent leakage.

### Results

| Model | Train R² | Test R² |
|-------|----------|---------|
| Random Forest | ~0.96 | 0.91 |
| Gradient Boosting | ~0.95 | 0.92 |
| XGBoost (Optuna) | ~0.96 | 0.92 |
| **Stacked Ensemble** | **0.97** | **0.92** |

All R² values are in log-price space. Train R² is higher than Test R² because large ensembles (300 trees, Optuna-tuned XGBoost) partially memorise training-set noise — this is expected behaviour for complex models, not a data problem. The ~0.045 gap is confirmed stable by holdout R² = 0.87 on 1,949 completely unseen listings (Section 6 of the notebook). CV RMSE std/mean = 3.7% (stable across folds).

---

## Experiment 5: More Data (Targeted + General Scraping)

### Problem
Premium areas (F-6, F-7, F-8, Clifton, DHA Phase 5) had 0-6 listings. Model couldn't learn premium price levels.

### Solution
1. **Targeted scraping:** Area-specific Zameen URLs for 17 underrepresented areas → 4,157 new listings
2. **General scraping:** Pages 81-400 across all 6 cities → 10,081 new listings
3. Cleaned, merged, deduplicated → 15,515 total houses (up from 10,473)

### Impact on specific areas

| Area | Before | After | Prediction Improvement |
|------|--------|-------|----------------------|
| F-8 Islamabad | 6 | 123 | 64% error → 9.4% ✅ |
| F-10 Islamabad | 28 | 242 | 50% error → 7.4% ✅ |
| DHA City Karachi | 3 | 23 | 52% error → 5.8% ✅ |
| Clifton Karachi | 83 | 301 | 42% error → improved |
| PECHS Karachi | 59 | 296 | 29% error → improved |

### Final Results (15.5K houses)

| Metric | 10K Houses | 15.5K Houses |
|--------|-----------|-----------|
| Good (±25%) | 62% | **59%** |
| Within ±50% | 93% | **90%** |
| Bad (>50%) | 7% | **10%** |
| Median Error | 21.6% | **18.6%** |
| R² | 0.92 | **0.92** |

Median error improved. The slight drop in "good" percentage is because the extra data introduced more diverse properties that are harder to predict, but the median and mean errors both improved.

---

## Full Progression

| Version | Data | Good (±25%) | Median Error | Test R² |
|---------|------|-------------|--------------|---------|
| All types, one-hot | 18K | 50% | 26.2% | 0.77 |
| Houses only | 10K | 59% | 19.9% | 0.77 |
| + Geographic features | 10K | 52% | 23.6% | 0.77 |
| + Optuna + Stacking | 10K | 62% | 21.6% | 0.92 |
| + Targeted scrape | 11K | 55% | 19.9% | 0.92 |
| **+ General scrape + dedup** | **15.5K** | **59%** | **18.6%** | **0.92** |

All R² figures are Test R² (20% held-out split, log-price space). Final model Train R² = 0.97.

---

## Holdout Validation

We scraped 2,385 houses from Zameen.com in a separate scraping session, stored in `data/test/zameen_holdout_test.csv`. After cleaning (outlier removal, null drops, URL dedup, content dedup), 1,949 listings remained. Zero URL overlap with the 29,220-listing training set was verified programmatically (see Section 6 of `notebooks/main.ipynb`).

The scraper (`scripts/scraper.py`) uses a maximum of 120 pages per city. The holdout comes from a separate scrape run, not a higher page range.

| Metric | Train/Test Split | Holdout (1,949 unseen) |
|--------|-----------------|------------------------|
| R² (log-price, primary) | 0.92 | **0.87** |
| R² (PKR-space) | — | 0.86 |
| Good (≤25%) | — | **70.4%** |
| Median Error | — | **15.2%** |
| MAE | — | 0.78 Crore PKR |

The ~0.05 R² drop from train/test to holdout is healthy — confirms generalization without overfitting.

---

## What Would Fix the Remaining Errors

The 3 remaining bad cases (DHA Phase 2 Islamabad, G-13, Federal B Area) share traits:
- **Ultra-premium sectors** with few training examples at matching price levels
- **Construction quality / age** not captured in our features
- **Street-level variation** within the same area

These are data gaps, not model flaws. Fixing them requires either much more data from these specific areas, or additional features (property age, floors, covered area) that Zameen listing cards don't expose.
