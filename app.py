"""
Pakistan Property Market Dashboard
===================================
Interactive Streamlit dashboard for exploring property prices,
market insights, and price predictions across 6 Pakistani cities.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import re

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pakistan Property Market",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label { color: #a0aec0 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e2e8f0 !important; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stMarkdown p { color: #cbd5e0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05); border-radius: 8px;
        color: #a0aec0; border: 1px solid rgba(255, 255, 255, 0.1); padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.3) !important;
        color: #ffffff !important; border-color: #6366f1 !important;
    }
    hr { border-color: rgba(255, 255, 255, 0.1); }
    .prediction-box {
        background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px; padding: 24px; text-align: center; backdrop-filter: blur(10px);
    }
    .prediction-price {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── SmoothedTargetEncoder (must be defined here so pickle can deserialize) ─────

class SmoothedTargetEncoder:
    def __init__(self, m=50):
        self.m = m; self.city_means = {}; self.loc_stats = {}; self.global_mean = None

    def fit(self, df, target_col='log_price'):
        self.global_mean = df[target_col].mean()
        self.city_means = df.groupby('city')[target_col].mean().to_dict()
        for (c, l), g in df.groupby(['city', 'location']):
            self.loc_stats[(c, l)] = (g[target_col].mean(), len(g))
        return self

    def transform(self, df):
        return np.array([self.encode_single(r['city'], r['location']) for _, r in df.iterrows()])

    def encode_single(self, city, loc):
        cm = self.city_means.get(city, self.global_mean)
        if (city, loc) in self.loc_stats:
            lm, n = self.loc_stats[(city, loc)]
            return (n * lm + self.m * cm) / (n + self.m)
        return cm


# ── Load Data & Model ──────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, 'data', 'processed', 'houses_cleaned.csv'))
    df['bedrooms'] = df['bedrooms'].fillna(0).astype(int)
    df['bathrooms'] = df['bathrooms'].fillna(0).astype(int)
    df['property_type'] = 'House'
    df['log_size'] = np.log1p(df['size_sqft'])
    return df

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, 'data', 'processed', 'model_artifacts.pkl'), 'rb') as f:
        return pickle.load(f)

df = load_data()
artifacts = load_model()
model = artifacts['model']
feature_cols = artifacts['feature_cols']
loc_encoder = artifacts.get('loc_encoder')

def _soc_type(loc):
    """Infer society_type from location string — mirrors notebook feature engineering."""
    ll = loc.lower()
    if 'dha' in ll or 'defence' in ll: return 'DHA'
    if 'bahria' in ll: return 'Bahria'
    if 'askari' in ll: return 'Askari'
    if re.search(r'\b[fgiedb]-\d', ll): return 'CDA_Sector'
    for k in ['park view','lake city','citi housing','eden','wapda','top city','capital smart',
              'naval anchorage','faisal town','paragon','izmir','valencia','central park','hayatabad']:
        if k in ll: return 'Private'
    for k in ['clifton','pechs','gulshan','nazimabad','model town','gulberg','cavalry','cantt',
              'saddar','johar town','garden town','sabzazar','scheme 33','malir','federal b']:
        if k in ll: return 'Established'
    return 'Other'

def _geo_features(city, loc):
    """Derive the 5 geographic features from city + location string."""
    ll = loc.lower() if loc else ''
    society = _soc_type(loc) if loc else 'Other'
    m = re.search(r'(?:phase|dha)\s*(\d+)', ll)
    dha_phase = int(m.group(1)) if m and ('dha' in ll or 'defence' in ll) else 0
    m2 = re.search(r'\b([fgiedb])-\d', ll)
    isb_tier = {'f':5,'e':4,'g':3,'h':3,'i':2,'d':1,'b':1}.get(m2.group(1), 0) if city == 'Islamabad' and m2 else 0
    is_premium = int(any([
        city in ['Lahore','Karachi'] and 'dha' in ll and bool(re.search(r'phase\s*[56]\b', ll)),
        city == 'Karachi' and 'dha' in ll and 'phase 8' in ll,
        city == 'Islamabad' and bool(re.search(r'\bf-[678]\b', ll)),
        city == 'Islamabad' and 'e-11' in ll,
        city == 'Karachi' and 'clifton' in ll,
        city == 'Lahore' and any(k in ll for k in ['gulberg iii', 'model town', 'cavalry']),
    ]))
    m3 = re.search(r'(?:phase|askari)\s*(\d+)', ll)
    phase_num = int(m3.group(1)) if m3 else 0
    return society, dha_phase, isb_tier, is_premium, phase_num

def make_feature_row(size, beds, baths, city, location=None):
    """Build a single-row DataFrame matching the model's feature_cols."""
    row = {c: 0 for c in feature_cols}
    row['size_sqft'] = size
    row['log_size'] = np.log1p(size)
    row['bedrooms'] = beds
    row['bathrooms'] = baths
    city_col = f'city_oh_{city}'
    if city_col in feature_cols:
        row[city_col] = 1
    society, dha_phase, isb_tier, is_premium, phase_num = _geo_features(city, location or '')
    soc_col = f'soc_{society}'
    if soc_col in feature_cols:
        row[soc_col] = 1
    row['dha_phase'] = dha_phase
    row['isb_sector_tier'] = isb_tier
    row['is_premium_area'] = is_premium
    row['phase_number'] = phase_num
    if 'location_encoded' in feature_cols and loc_encoder:
        if location:
            row['location_encoded'] = loc_encoder.encode_single(city, location)
        else:
            row['location_encoded'] = loc_encoder.city_means.get(city, loc_encoder.global_mean)
    return pd.DataFrame([row])[feature_cols]

def make_feature_df(dataframe):
    """Build model-ready features for a full DataFrame."""
    rows = []
    for _, r in dataframe.iterrows():
        row = {c: 0 for c in feature_cols}
        row['size_sqft'] = r['size_sqft']
        row['log_size'] = np.log1p(r['size_sqft'])
        row['bedrooms'] = r['bedrooms']
        row['bathrooms'] = r['bathrooms']
        row['dha_phase'] = r.get('dha_phase', 0)
        row['isb_sector_tier'] = r.get('isb_sector_tier', 0)
        row['is_premium_area'] = r.get('is_premium_area', 0)
        row['phase_number'] = r.get('phase_number', 0)
        city_col = f"city_oh_{r['city']}"
        if city_col in feature_cols:
            row[city_col] = 1
        soc_col = f"soc_{r.get('society_type', 'Other')}"
        if soc_col in feature_cols:
            row[soc_col] = 1
        if 'location_encoded' in feature_cols and loc_encoder:
            row['location_encoded'] = loc_encoder.encode_single(r['city'], r['location'])
        rows.append(row)
    return pd.DataFrame(rows)[feature_cols]

PLOTLY_TEMPLATE = "plotly_dark"

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🏠 Filters")
    st.markdown("---")
    selected_cities = st.multiselect("Cities", sorted(df['city'].unique()), default=sorted(df['city'].unique()))
    price_range = st.slider("Price Range (Crore PKR)", 0.0, float(df['price_pkr'].max() / 1e7), (0.0, float(df['price_pkr'].max() / 1e7)), step=0.5)
    size_range = st.slider("Size Range (sqft)", 0, int(df['size_sqft'].max()), (0, int(df['size_sqft'].max())), step=100)
    st.markdown("---")
    st.markdown(f"<p style='color: #64748b; font-size: 0.75rem;'>Data: Zameen.com | {len(df):,} listings<br>DS 401 — NUST SEECS, Spring 2026</p>", unsafe_allow_html=True)

mask = (
    df['city'].isin(selected_cities)
    & (df['price_pkr'] >= price_range[0] * 1e7) & (df['price_pkr'] <= price_range[1] * 1e7)
    & (df['size_sqft'] >= size_range[0]) & (df['size_sqft'] <= size_range[1])
)
filtered = df[mask]

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("# 🏠 Pakistan Property Market Dashboard")
st.markdown("*Exploring property prices, market insights, and AI-powered price predictions across 6 major cities*")
st.markdown("---")

# ── KPI Cards ──────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Listings", f"{len(filtered):,}")
with k2: st.metric("Median Price", f"{filtered['price_pkr'].median()/1e7:.1f} Cr")
with k3: st.metric("Median PKR/sqft", f"{filtered['price_per_sqft'].median():,.0f}")
with k4: st.metric("Median Size", f"{filtered['size_sqft'].median():,.0f} sqft")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab5, tab3, tab4 = st.tabs(["📊 Market Overview", "🗺️ City Comparison", "🏘️ Area Analysis", "🔍 Overpriced vs Undervalued", "🤖 Price Predictor"])

# ── Tab 1: Market Overview ─────────────────────────────────────────────────────

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered, x='price_pkr', nbins=50, color='city',
                           color_discrete_sequence=px.colors.qualitative.Set2, template=PLOTLY_TEMPLATE,
                           barmode='overlay', opacity=0.75,
                           labels={'price_pkr': 'Price (PKR)', 'city': 'City'})
        fig.update_layout(title='House Price Distribution by City', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_counts = filtered['city'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=city_counts.index, values=city_counts.values, hole=0.55,
                                      marker=dict(colors=px.colors.qualitative.Set2),
                                      textfont=dict(color='white', size=14))])
        fig.update_layout(title='Listings by City', template=PLOTLY_TEMPLATE,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=400)
        st.plotly_chart(fig, use_container_width=True)

    sample = filtered.sample(n=min(5000, len(filtered)), random_state=42)
    fig = px.scatter(sample, x='size_sqft', y='price_pkr', color='city',
                     color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.5, template=PLOTLY_TEMPLATE,
                     labels={'size_sqft': 'Size (sqft)', 'price_pkr': 'Price (PKR)', 'city': 'City'},
                     hover_data=['bedrooms', 'price_per_sqft', 'location'])
    fig.update_layout(title='House Size vs Price by City', plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: City Comparison ─────────────────────────────────────────────────────

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        city_ppsf = filtered.groupby('city')['price_per_sqft'].median().sort_values()
        fig = go.Figure(go.Bar(x=city_ppsf.values, y=city_ppsf.index, orientation='h',
                                marker=dict(color=city_ppsf.values, colorscale='Viridis', line=dict(width=0)),
                                text=[f'{v:,.0f}' for v in city_ppsf.values], textposition='outside',
                                textfont=dict(color='#e2e8f0', size=13)))
        fig.update_layout(title='Median Price per Sqft by City', xaxis_title='PKR / sqft', template=PLOTLY_TEMPLATE,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_counts = filtered['city'].value_counts().sort_values()
        fig = go.Figure(go.Bar(x=city_counts.values, y=city_counts.index, orientation='h',
                                marker=dict(color=city_counts.values, colorscale='Plasma', line=dict(width=0)),
                                text=[f'{v:,}' for v in city_counts.values], textposition='outside',
                                textfont=dict(color='#e2e8f0', size=13)))
        fig.update_layout(title='Number of Listings by City', xaxis_title='Count', template=PLOTLY_TEMPLATE,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=400)
        st.plotly_chart(fig, use_container_width=True)

    bed_df = filtered.copy()
    bed_df['bedroom_group'] = pd.cut(bed_df['bedrooms'], bins=[0, 3, 5, 100],
                                      labels=['1–3 Bed', '4–5 Bed', '6+ Bed'])
    pivot = bed_df.groupby(['city', 'bedroom_group'], observed=True)['price_per_sqft'].median().reset_index()
    fig = px.bar(pivot, x='city', y='price_per_sqft', color='bedroom_group', barmode='group',
                 color_discrete_sequence=['#6366f1', '#10b981', '#f43f5e'], template=PLOTLY_TEMPLATE,
                 labels={'price_per_sqft': 'Median PKR/sqft', 'city': 'City', 'bedroom_group': 'Bedrooms'})
    fig.update_layout(title='Median Price per Sqft: City × Bedroom Group', plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=450)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: Area Analysis ───────────────────────────────────────────────────────

with tab5:
    st.markdown("### Area-wise Price Analysis")
    st.markdown("Explore how property prices vary across different areas within each city. Areas with fewer than 20 listings are grouped as 'Other'.")

    area_city = st.selectbox("Select City for Area Analysis", sorted(df['city'].unique()), key='area_city')

    city_data = filtered[filtered['city'] == area_city]

    # Group areas: 20+ listings get their own name, rest → "Other"
    loc_counts_city = city_data['location'].value_counts()
    named_areas = loc_counts_city[loc_counts_city >= 20].index.tolist()

    city_data = city_data.copy()
    city_data['area'] = city_data['location'].apply(lambda x: x if x in named_areas else 'Other')

    # Area stats
    area_stats = city_data.groupby('area').agg(
        listings=('price_pkr', 'count'),
        median_price=('price_pkr', 'median'),
        median_ppsf=('price_per_sqft', 'median'),
        median_size=('size_sqft', 'median'),
    ).sort_values('median_ppsf', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # Price per sqft by area
        plot_data = area_stats[area_stats.index != 'Other'].head(20).sort_values('median_ppsf')
        fig = go.Figure(go.Bar(
            x=plot_data['median_ppsf'].values,
            y=plot_data.index,
            orientation='h',
            marker=dict(color=plot_data['median_ppsf'].values, colorscale='Viridis', line=dict(width=0)),
            text=[f'{v:,.0f}' for v in plot_data['median_ppsf'].values],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
        ))
        fig.update_layout(
            title=f'Median Price/Sqft by Area — Houses in {area_city}',
            xaxis_title='PKR / sqft', template=PLOTLY_TEMPLATE,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=max(400, len(plot_data) * 30 + 100),
            margin=dict(l=250),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Median price by area
        plot_data2 = area_stats[area_stats.index != 'Other'].head(20).sort_values('median_price')
        fig = go.Figure(go.Bar(
            x=plot_data2['median_price'].values / 1e7,
            y=plot_data2.index,
            orientation='h',
            marker=dict(color=plot_data2['median_price'].values, colorscale='Plasma', line=dict(width=0)),
            text=[f'{v/1e7:.1f} Cr' for v in plot_data2['median_price'].values],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
        ))
        fig.update_layout(
            title=f'Median Total Price by Area — Houses in {area_city}',
            xaxis_title='Price (Crore PKR)', template=PLOTLY_TEMPLATE,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=max(400, len(plot_data2) * 30 + 100),
            margin=dict(l=250),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown(f"### Detailed Area Breakdown — Houses in {area_city}")
    table_data = area_stats.copy()
    table_data['median_price'] = (table_data['median_price'] / 1e7).round(2).astype(str) + ' Cr'
    table_data['median_ppsf'] = table_data['median_ppsf'].round(0).astype(int).apply(lambda x: f'{x:,}')
    table_data['median_size'] = table_data['median_size'].round(0).astype(int).apply(lambda x: f'{x:,}')
    table_data.columns = ['Listings', 'Median Price', 'Median PKR/sqft', 'Median Size (sqft)']
    table_data = table_data.sort_values('Listings', ascending=False)
    st.dataframe(table_data, use_container_width=True)

# ── Tab 3: Overpriced vs Undervalued ───────────────────────────────────────────

with tab3:
    X_pred = make_feature_df(filtered)
    predicted_log = model.predict(X_pred)
    predicted_pkr = np.expm1(predicted_log)

    filt_analysis = filtered.copy()
    filt_analysis['predicted_pkr'] = predicted_pkr
    filt_analysis['price_ratio'] = filt_analysis['price_pkr'] / filt_analysis['predicted_pkr']
    filt_analysis['status'] = filt_analysis['price_ratio'].apply(
        lambda r: 'Overpriced' if r > 1.20 else ('Undervalued' if r < 0.80 else 'Fair'))

    s1, s2, s3 = st.columns(3)
    status_counts = filt_analysis['status'].value_counts()
    with s1:
        ov = status_counts.get('Overpriced', 0)
        st.metric("🔴 Overpriced", f"{ov:,}", f"{ov/len(filt_analysis)*100:.1f}%")
    with s2:
        fair = status_counts.get('Fair', 0)
        st.metric("⚪ Fair", f"{fair:,}", f"{fair/len(filt_analysis)*100:.1f}%")
    with s3:
        uv = status_counts.get('Undervalued', 0)
        st.metric("🟢 Undervalued", f"{uv:,}", f"{uv/len(filt_analysis)*100:.1f}%")

    st.markdown("")
    col1, col2 = st.columns(2)
    status_colors_map = {'Overpriced': '#ef4444', 'Fair': '#64748b', 'Undervalued': '#10b981'}

    with col1:
        fig = px.scatter(filt_analysis.sample(n=min(5000, len(filt_analysis)), random_state=42),
                         x='predicted_pkr', y='price_pkr', color='status', color_discrete_map=status_colors_map,
                         opacity=0.5, template=PLOTLY_TEMPLATE,
                         labels={'predicted_pkr': 'Predicted Price (PKR)', 'price_pkr': 'Actual Price (PKR)'},
                         hover_data=['city', 'size_sqft', 'location'])
        max_val = max(filt_analysis['price_pkr'].max(), filt_analysis['predicted_pkr'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                  line=dict(dash='dash', color='white', width=1), name='Fair Price Line'))
        fig.update_layout(title='Actual vs Predicted Price', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_status = filt_analysis.groupby(['city', 'status']).size().reset_index(name='count')
        fig = px.bar(city_status, x='city', y='count', color='status', color_discrete_map=status_colors_map,
                     barmode='group', template=PLOTLY_TEMPLATE, labels={'count': 'Count', 'city': 'City'})
        fig.update_layout(title='Pricing Status by City', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🟢 Top Undervalued Properties (Best Deals)")
    undervalued = filt_analysis[filt_analysis['status'] == 'Undervalued'].copy()
    undervalued['savings'] = undervalued['predicted_pkr'] - undervalued['price_pkr']
    top_deals = undervalued.nlargest(10, 'savings')[
        ['city', 'location', 'size_sqft', 'bedrooms', 'price_pkr', 'predicted_pkr', 'savings']
    ].copy()
    top_deals['price_pkr'] = (top_deals['price_pkr'] / 1e7).round(2).astype(str) + ' Cr'
    top_deals['predicted_pkr'] = (top_deals['predicted_pkr'] / 1e7).round(2).astype(str) + ' Cr'
    top_deals['savings'] = (top_deals['savings'] / 1e5).round(0).astype(int).astype(str) + ' Lakh'
    top_deals.columns = ['City', 'Location', 'Size (sqft)', 'Beds', 'Listed Price', 'Fair Value', 'Potential Savings']
    st.dataframe(top_deals, use_container_width=True, hide_index=True)

# ── Tab 4: Price Predictor ─────────────────────────────────────────────────────

with tab4:
    st.markdown("### Enter property details to get an AI-powered price estimate")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        pred_city = st.selectbox("City", sorted(df['city'].unique()), index=1)

    with col2:
        pred_size = st.number_input("Size (sqft)", min_value=50, max_value=100000, value=2250, step=50)
        pred_beds = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)

    with col3:
        loc_counts = df[df['city'] == pred_city]['location'].value_counts()
        city_locs = loc_counts[loc_counts >= 10].index.tolist()
        loc_options = ['(General / Other)'] + sorted(city_locs)
        pred_loc = st.selectbox("Area (optional)", loc_options)
        if pred_loc == '(General / Other)':
            pred_loc = None

        pred_baths = st.number_input("Bathrooms", min_value=0, max_value=10, value=3, step=1)

    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        features = make_feature_row(pred_size, pred_beds, pred_baths, pred_city, pred_loc)
        pred_log = model.predict(features)[0]
        pred_pkr = np.expm1(pred_log)

        similar = df[
            (df['city'] == pred_city)
            & (df['size_sqft'].between(pred_size * 0.8, pred_size * 1.2))
        ]
        median_similar = similar['price_pkr'].median() if len(similar) > 5 else None

        st.markdown("---")

        price_str = f"{pred_pkr / 1e7:.2f} Crore PKR" if pred_pkr >= 1e7 else f"{pred_pkr / 1e5:.0f} Lakh PKR"
        ppsf = pred_pkr / pred_size

        loc_display = f" in {pred_loc}" if pred_loc else ""
        beds_display = f"{pred_beds} bed / {pred_baths} bath"

        st.markdown(f"""
        <div class="prediction-box">
            <p style="color: #a0aec0; font-size: 1rem; margin: 0;">Estimated Fair Market Value</p>
            <p class="prediction-price">{price_str}</p>
            <p style="color: #a0aec0; font-size: 0.95rem; margin: 0;">
                {pred_size:,} sqft House in {pred_city}{loc_display} &bull; {beds_display}
                &bull; {ppsf:,.0f} PKR/sqft
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        c1, c2, c3 = st.columns(3)
        with c1:
            city_median = df[df['city'] == pred_city]['price_per_sqft'].median()
            st.metric(f"{pred_city} Median PKR/sqft", f"{city_median:,.0f}")
        with c2:
            if median_similar is not None:
                st.metric("Similar Properties Median", f"{median_similar/1e7:.2f} Cr")
            else:
                st.metric("Similar Properties", "Insufficient data")
        with c3:
            if median_similar is not None:
                diff = ((pred_pkr - median_similar) / median_similar) * 100
                st.metric("vs Market Median", f"{diff:+.1f}%")
            else:
                st.metric("vs Market Median", "N/A")
