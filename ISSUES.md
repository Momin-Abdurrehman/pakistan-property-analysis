# Issues & Resolutions Log

A detailed record of every significant issue we encountered during the development of this project, and how each was resolved.

---

## 1. Data Collection Issues

### 1.1 Cloudflare Anti-Bot Protection
**Issue:** Zameen.com uses Cloudflare protection. Simple `requests`-based HTTP scraping was immediately blocked — responses returned Cloudflare challenge pages instead of listing data.

**Resolution:** Switched to **Playwright** (headless Chromium browser), which executes JavaScript and passes Cloudflare checks. This mimics a real browser session and successfully loads listing pages.

---

### 1.2 Incorrect City URL Codes
**Issue:** We initially assumed Zameen.com URL codes mapped intuitively (e.g., Lahore=2, Karachi=1). In reality, `Lahore-2` resolved to **Karachi**, not Lahore. The city ID numbers are internal Zameen identifiers, not sequential.

**Resolution:** Systematically tested URL patterns by loading pages and checking the resolved `page.url` and `page.title()`. Discovered the correct mapping:
- Lahore = 1, Karachi = 2, Islamabad = 3, Faisalabad = 16, Peshawar = 17, Rawalpindi = 41

---

### 1.3 Single Property Type Was Insufficient
**Issue:** Initially we only scraped "Homes" (Houses), resulting in 9,250 listings. The `property_type` column had zero variation — all entries were "House". This meant property type couldn't be used as a feature, and the analysis lacked depth.

**Resolution:** Expanded scraping to include **Flats/Apartments** and **Plots**. This tripled the dataset to 24,373 listings and added meaningful property type variation for both EDA and modeling.

---

### 1.4 Duplicate Listings Across Scraper Runs
**Issue:** Multiple scraper processes ran concurrently (background tasks), and Zameen.com re-shows popular listings across pagination pages. This resulted in ~1,161 duplicate rows in the merged CSV.

**Resolution:** Two-stage deduplication:
1. URL-based dedup (for rows with non-null URLs) — removes exact re-scraped listings
2. Content-based dedup on `title + city + price + size + property_type` — catches duplicates without URLs

---

### 1.5 Background Scraper Working Directory Issue
**Issue:** The first background scraper process produced no output — the CSV file was never created. The background process used a relative path (`data/raw/zameen_raw.csv`) but ran from a different working directory.

**Resolution:** Switched to absolute paths (`/Users/.../data/raw/zameen_raw.csv`) for all file operations in the scraper. Ran the main scrape in foreground to ensure reliability.

---

## 2. Data Cleaning Issues

### 2.1 Mixed Size Units (4 Different Systems)
**Issue:** Property sizes used 4 different unit systems:
- Marla (10,641 listings) — common for houses in Punjab/KPK
- Kanal (4,218 listings) — larger properties
- sqft (5,162 listings) — used by Flats
- Sq. Yd. (4,352 listings) — common in Karachi/Sindh

Direct comparison was impossible without standardization.

**Resolution:** Converted all sizes to square feet using documented conversion rates:
- Marla × 225, Kanal × 4,500, Sq. Yd. × 9, sqft as-is

---

### 2.2 City-Specific Marla Variation
**Issue:** The Marla unit varies by city in Pakistan government records (Islamabad ≈ 250 sqft, Lahore ≈ 225 sqft, Karachi ≈ 240 sqft). Using city-specific rates would be more accurate but adds complexity.

**Resolution:** Used a flat 225 sqft/Marla universally. Rationale: Zameen.com internally standardizes on 225 sqft per Marla in their listings. Using one consistent value matches the source data and avoids introducing artificial regional variance. Documented as a limitation.

---

### 2.3 Commas in Size Numbers
**Issue:** 3,237 size entries contained commas (e.g., "1,591 Sq. Yd.", "1,502 sqft"). The `float()` parser failed on these strings.

**Resolution:** Added `.replace(',', '')` before parsing the numeric value in the `parse_size_to_sqft()` function.

---

### 2.4 Zero-Size Entries
**Issue:** 2 entries had "0 Marla" — resulting in 0 sqft after conversion. These are data entry errors on Zameen.com.

**Resolution:** Dropped rows where `size_sqft == 0` after conversion.

---

### 2.5 Plot Bedrooms/Bathrooms Are Legitimately Null
**Issue:** All 5,945 Plot listings had null bedrooms and bathrooms. Initially this looked like a data quality problem, but plots are empty land — they genuinely have no rooms.

**Resolution:** Two-part strategy:
- For Plots: filled with 0 (semantically correct — zero rooms)
- For Houses/Flats with null beds/baths (~990 rows, 5-7%): dropped as incomplete data

---

### 2.6 Five Dirty Rows Survived Outlier Removal
**Issue:** IQR-based outlier removal didn't catch 5 entries that were within IQR bounds but were clearly bad data:
- 36 sqft "House" in Karachi (that's a closet, not a house)
- 90 sqft "House" in Islamabad at 13 Crore (obvious misparse)
- 117 sqft House at 376K PKR/sqft (likely "117 Sq. Yd." mislabeled as sqft)
- 1.4 Lakh for a 1 Kanal house in DHA Islamabad (placeholder price)
- 1.0 Lakh for a plot in Islamabad (not a real listing)

**Resolution:** Added domain-based sanity filters after IQR outlier removal:
- Houses/Flats must be >= 200 sqft
- Plots must be >= 100 sqft
- All properties must be >= 5 Lakh PKR (500,000)

---

## 3. EDA Issues

### 3.1 Correlation Matrix Included Label-Encoded Categoricals
**Issue:** The initial correlation heatmap included `city_encoded` and `property_type_encoded` (label-encoded). Pearson correlation on label-encoded categoricals is statistically meaningless — it implies Faisalabad(0) < Islamabad(1) < Karachi(2), which is a false ordinal relationship.

**Resolution:** Removed label-encoded categoricals from the correlation matrix. Kept only genuine numeric features: `price_pkr`, `size_sqft`, `bedrooms`, `bathrooms`, `price_per_sqft`, `days_since_listed`. Added a note explaining why categorical effects are analyzed via box plots instead.

---

### 3.2 Pair Plot Performance
**Issue:** Running a pair plot on the full 18,443 rows with all features was extremely slow (matplotlib rendering thousands of scatter plots).

**Resolution:** Used a random sample of 4,000 rows (`df.sample(n=4000)`) for the pair plot. This maintains visual patterns while keeping render time reasonable.

---

## 4. Modeling Issues

### 4.1 House and Plot Predicted at Nearly Same Price
**Issue:** A 10 Marla house was predicted at 1.88 Crore while a 10 Marla plot was predicted at 1.84 Crore — virtually identical. In reality, houses cost 3-4x more than plots of the same size because of construction costs.

**Root cause:** Two problems:
1. **Label encoding:** `property_type_encoded` (0, 1, 2) had only 1.5% feature importance — the model barely used it
2. **App UI bug:** When selecting "Plot", the bedrooms/bathrooms inputs stayed at 3/3 instead of resetting to 0. The model saw "something with 3 bathrooms" and predicted a House-like price.

**Resolution:**
1. Switched from label encoding to **one-hot encoding** for property type and city
2. Added top 50 locations as one-hot features (62 total features)
3. Fixed the app to auto-zero beds/baths when "Plot" is selected
4. Result: House = 5.46 Cr vs Plot = 1.39 Cr (realistic 4x gap)

---

### 4.2 No Feature Scaling for Linear Regression
**Issue:** All features were fed raw into models. While tree models are scale-invariant, Linear Regression is affected by feature scale — `size_sqft` (~2000) dominated over `bathrooms` (~3) simply because of magnitude.

**Resolution:** Added `StandardScaler` on numeric features (`size_sqft`, `bedrooms`, `bathrooms`) for Linear Regression only. Tree models continue to use unscaled features. Added justification in the notebook.

---

### 4.3 Cross-City Location Feature Bleed
**Issue:** Location features like `loc_DHA Phase 6` fire for both Lahore's DHA Phase 6 and Karachi's DHA Phase 6 — completely different neighborhoods with different prices. The model learned a blended price effect dominated by whichever city has more listings, causing city-level price rankings to sometimes invert.

**Resolution:** Documented as a known limitation. The proper fix (interaction features like `loc_Lahore_DHA_Phase_6`) would expand the feature space from 62 to 300+ features, risking overfitting with our dataset size. This trade-off is explicitly discussed in the Limitations section.

---

### 4.4 Bathrooms Dominated Feature Importance (46%)
**Issue:** `bathrooms` had 46% feature importance, which seems counterintuitive — bathrooms shouldn't be the primary driver of property price.

**Explanation:** Bathrooms acts as a **proxy variable** for property development level. A property with 0 bathrooms is empty land (Plot), 1-2 bathrooms is a small flat, and 5-6+ bathrooms is a large constructed house. The model uses bathrooms as a cheap signal for distinguishing property types and development level, even beyond the explicit property_type feature.

**Resolution:** No code fix needed — this is an expected behavior of tree models. Documented in the notebook summary (§4.9) with the explanation.

---

### 4.5 Scraper Code Not In Notebook
**Issue:** The rubric requires "fully reproducible code to acquire the data." The notebook only had `pd.read_csv()` with a markdown description of the scraping method. No runnable acquisition code was present.

**Resolution:** Added a full scraping code cell in Section 1.2 with a `RUN_SCRAPER = False` flag. When `False`, it loads from the saved CSV. When `True`, it runs the complete Playwright scraper. This satisfies the rubric while keeping the notebook fast by default.

---

### 4.6 Model Pickle Too Large for Git (80MB)
**Issue:** The saved `model_artifacts.pkl` (Random Forest with 200 trees, 62 features) was 80MB — too large for a Git repository.

**Resolution:** Added `data/processed/model_artifacts.pkl` to `.gitignore`. The model is automatically regenerated when Section 4 of the notebook is run. Added a note in the README explaining this.

---

### 4.7 No Hyperparameter Tuning
**Issue:** Random Forest and Gradient Boosting used hardcoded hyperparameters (`n_estimators=200`, `max_depth=5`). While the rubric doesn't strictly require tuning, it strengthens the modeling section.

**Resolution:** Added a `GridSearchCV` cell (§4.6) that tunes Random Forest over `n_estimators`, `max_depth`, and `min_samples_leaf`. The tuned model is compared against the defaults and the best overall model is selected for diagnostics.

---

## 5. Dashboard Issues

### 5.1 Plot Type Didn't Disable Bedrooms/Bathrooms
**Issue:** In the Price Predictor tab, selecting "Plot" as property type kept the bedrooms and bathrooms inputs active. Users could enter 3 beds / 3 baths for a plot, causing the model to predict house-like prices for empty land.

**Resolution:** Added conditional logic: when `pred_type == 'Plot'`, bedrooms and bathrooms inputs are replaced with info messages ("Plots have no bedrooms") and values are forced to 0.

---

### 5.2 Feature Mismatch Between App and Model
**Issue:** The saved model expected 62 one-hot features, but the app initially tried to pass 5 label-encoded features. This would crash the prediction.

**Resolution:** Rewrote the app to use `make_feature_row()` and `make_feature_df()` helper functions that build 62-column DataFrames from scratch, matching the model's expected input exactly. The feature list is loaded directly from the pickle artifacts.

---

### 5.3 No Area-Wise Analysis
**Issue:** The dashboard showed city-level comparisons but no neighborhood-level breakdown. Within a single city like Islamabad, DHA Phase 2 costs 4x more per sqft than B-17 — city averages hide this.

**Resolution:** Added a dedicated **🏘️ Area Analysis** tab with:
- City + property type selectors
- Median price/sqft by area (horizontal bar chart)
- Median total price by area
- Detailed table with all area statistics
- Areas with <20 listings grouped as "Other"
