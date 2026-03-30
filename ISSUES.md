# Issues & Resolutions Log

A detailed record of every significant issue we encountered during development, and how each was resolved.

---

## 1. Data Collection

### 1.1 Cloudflare Anti-Bot Protection
**Issue:** Zameen.com uses Cloudflare protection. Simple `requests`-based scraping returned challenge pages instead of listing data.

**Resolution:** Switched to **Playwright** (headless Chromium), which executes JavaScript and passes Cloudflare checks.

### 1.2 Incorrect City URL Codes
**Issue:** Zameen URL codes don't map intuitively — `Lahore-2` resolved to Karachi. City IDs are internal Zameen identifiers.

**Resolution:** Systematically tested URL patterns and discovered the correct mapping: Lahore=1, Karachi=2, Islamabad=3, Faisalabad=16, Peshawar=17, Rawalpindi=41.

### 1.3 Insufficient Data in Premium Areas
**Issue:** Initial scraping (pages 1-80 per city) yielded only 1-6 listings for premium Islamabad sectors (F-6, F-7, F-8). These were our worst prediction errors.

**Resolution:** Scraped area-specific Zameen URLs targeting 17 underrepresented areas, then scraped pages 81-400 across all cities. Total dataset grew from 10K to 14K+ houses.

### 1.4 Duplicate Listings Across Scraper Runs
**Issue:** Multiple scraping sessions and Zameen re-showing popular listings created ~5,000+ duplicate rows.

**Resolution:** Two-stage deduplication: URL-based for rows with URLs, content-based (location+city+price+size+bedrooms) for the rest.

---

## 2. Data Cleaning

### 2.1 Mixed Size Units (4 Systems)
**Issue:** Property sizes used Marla (10,641), Kanal (4,218), sqft (5,162), and Sq. Yd. (4,352) — impossible to compare directly.

**Resolution:** Standardized everything to square feet: Marla×225, Kanal×4,500, Sq.Yd.×9, sqft as-is.

### 2.2 Marla Varies by City
**Issue:** Government records show Islamabad≈250, Lahore≈225, Karachi≈240 sqft per Marla. Using city-specific rates adds complexity.

**Resolution:** Used flat 225 sqft/Marla. Zameen.com internally standardizes on 225. Documented as a limitation.

### 2.3 Five Dirty Rows Survived IQR Outlier Removal
**Issue:** IQR missed 5 data-entry errors: 36 sqft "house", 90 sqft house at 13 Crore, 1.4 Lakh for 1 Kanal in DHA.

**Resolution:** Added domain sanity filters: Houses/Flats ≥ 200 sqft, all properties ≥ 5 Lakh PKR.

### 2.4 Plot Bedrooms/Bathrooms Are Null
**Issue:** All 6,000+ Plot listings had null bedrooms/bathrooms — plots are empty land.

**Resolution:** In the houses-only approach, we removed Plots and Flats entirely. This eliminated the "bathrooms as proxy for property type" problem where 46% of model importance went to bathrooms.

---

## 3. EDA

### 3.1 Correlation Matrix Had Label-Encoded Categoricals
**Issue:** Initial heatmap included `city_encoded` and `property_type_encoded` — Pearson correlation on label-encoded categoricals implies false ordinal relationships.

**Resolution:** Removed from correlation matrix. Categorical effects analyzed via box plots instead.

---

## 4. Modeling

### 4.1 House and Plot Predicted at Same Price
**Issue:** 10 Marla house predicted at 1.88 Cr, 10 Marla plot at 1.84 Cr. Houses should be 3-4x more expensive.

**Root cause:** Label encoding gave property_type only 1.5% feature importance. Bathrooms (46%) acted as the real type discriminator — but the app didn't reset beds/baths to 0 for plots.

**Resolution:** Switched to houses-only modeling. Eliminated the cross-type confusion entirely. Bathrooms now reflects actual house quality instead of "is this built or empty land."

### 4.2 Cross-City Location Feature Bleed
**Issue:** Shared location features (e.g., `loc_DHA Phase 6`) blended prices from Lahore and Karachi DHA — different neighborhoods, different prices.

**Resolution:** Tested three approaches: shared one-hot, city-prefixed one-hot, and smoothed Bayesian target encoding. Target encoding performed best on individual cases. Documented in MODEL_EXPERIMENTS.md.

### 4.3 Premium Sectors Had Too Few Listings
**Issue:** F-8 Islamabad (6 listings), F-6 (1 listing), F-7 (0 listings). Model couldn't learn premium price levels.

**Resolution:** Targeted scraping of area-specific Zameen URLs. F-8 grew from 6→123 listings, F-10 from 28→242. F-8 prediction improved from 64% error to 9.4%.

### 4.4 Derived Geographic Features from Domain Knowledge
**Issue:** The model had no signal about the type of development (DHA vs Bahria vs government sector) or premium status.

**Resolution:** Extracted 5 features from location strings using Pakistan property market domain knowledge:
- `society_type`: DHA/Bahria/Askari/CDA_Sector/Private/Established/Other
- `dha_phase`: phase number for DHA properties
- `isb_sector_tier`: F=5 > E=4 > G=3 > I=2 > B,D=1 for Islamabad CDA sectors
- `is_premium_area`: binary flag for known premium locations
- `phase_number`: generic phase/block number

### 4.5 Model Tuning with Optuna + Stacking
**Issue:** Default hyperparameters left performance on the table.

**Resolution:** Used Optuna Bayesian optimization (200 trials) to tune XGBoost. Built a stacked ensemble (RF + GB + XGBoost → Ridge meta-learner). R² improved from 0.77 to 0.93.

---

## 5. Dashboard

### 5.1 Plot Type Didn't Disable Bedrooms/Bathrooms
**Issue:** Selecting "Plot" in Price Predictor kept beds/baths inputs active, causing house-like price predictions for empty land.

**Resolution:** Conditional logic: when Plot is selected, inputs replaced with info messages and values forced to 0.

### 5.2 No Area-Wise Analysis
**Issue:** Dashboard showed only city-level comparisons. DHA Phase 2 costs 4x more than B-17 within the same city.

**Resolution:** Added a dedicated Area Analysis tab with per-city, per-area price breakdowns.
