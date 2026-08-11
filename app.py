import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rainfall Prediction ML",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ---- Global ---- */
        /* ---- Fix: expander content text color ---- */
    [data-testid="stExpander"] {
        background: white !important;
        border: 1px solid #dcd9d5 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div,
    [data-testid="stExpander"] label {
        color: #28251d !important;
    }

    /* ---- Fix: all body text + headings on light background ---- */
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li,
    .main [data-testid="stMarkdownContainer"] span,
    .main [data-testid="stMarkdownContainer"] div,
    .main [data-testid="stMarkdownContainer"] h1,
    .main [data-testid="stMarkdownContainer"] h2,
    .main [data-testid="stMarkdownContainer"] h3,
    .main p, .main span, .main div, .main label {
        color: #28251d !important;
    }

    /* ---- Fix: st.title() / st.header() / st.subheader() ---- */
    .main h1, .main h2, .main h3, .main h4 {
        color: #0f3638 !important;
    }

    /* ---- Fix: Streamlit's native title element ---- */
    [data-testid="stHeading"],
    [data-testid="stHeadingWithActionElements"] {
        color: #0f3638 !important;
    }
    [data-testid="stHeading"] *,
    [data-testid="stHeadingWithActionElements"] * {
        color: #0f3638 !important;
    }

    /* ---- Fix: tab label text ---- */
    .stTabs [data-baseweb="tab"] span {
        color: #28251d !important;
    }
    .stTabs [aria-selected="true"] span {
        color: white !important;
    }

    /* ---- Fix: selectbox text ---- */
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: #28251d !important;
    }

    /* ---- Fix: section-title inside columns ---- */
    .section-title {
        color: #0f3638 !important;
    }

    /* ---- Fix: metric widget labels and values on light bg ---- */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #28251d !important;
    }
    [data-testid="stAppViewContainer"] {
        background: #f7f6f2;
    }
    [data-testid="stSidebar"] {
        background: #1c1b19 !important;
    }
    [data-testid="stSidebar"] * {
        color: #cdccca !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #cdccca !important;
    }

    /* ---- Header ---- */
    .hero-header {
        background: linear-gradient(135deg, #01696f 0%, #0f3638 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-header h1 { color: white; font-size: 2rem; margin: 0; }
    .hero-header p  { color: rgba(255,255,255,0.82); margin: 0.4rem 0 0 0; font-size: 1rem; }

    /* ---- Metric Cards ---- */
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        flex: 1 1 150px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 3px solid #01696f;
    }
    .metric-card .label { font-size: 0.78rem; color: #7a7974; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f3638; margin-top: 0.2rem; }
    .metric-card .sub   { font-size: 0.8rem; color: #437a22; margin-top: 0.1rem; }

    /* ---- Section headings ---- */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f3638;
        border-left: 4px solid #01696f;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ---- Tab pills ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #e6e4df;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-size: 0.87rem;
        font-weight: 600;
        color: #28251d;
    }
    .stTabs [aria-selected="true"] {
        background: #01696f !important;
        color: white !important;
    }

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* ---- Badge ---- */
    .badge {
        display: inline-block;
        background: #cedcd8;
        color: #0f3638;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .badge-gold { background: #e9e0c6; color: #8a5b00; }
</style>
""", unsafe_allow_html=True)

# ── Helper: load CSVs ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rainfall_prediction_model.pkl"
)


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


model = load_model()

if model is None:
    st.error("❌ Rainfall prediction model not found.")
else:
    st.success("✅ Rainfall prediction model loaded successfully.")
    
@st.cache_data
def load_cleaned_data():
    path = os.path.join(OUTPUTS, "cleaned_rainfall_dataset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["DATE"])
        return df
    return None

@st.cache_data
def load_predictions():
    path = os.path.join(OUTPUTS, "rainfall_predictions_2021_2025.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    return None

df = load_cleaned_data()
pred_df = load_predictions()

# ── Model results (hardcoded from pipeline outputs) ───────────
MODEL_RESULTS = pd.DataFrame([
    {"Model": "XGBoost",           "R² Train": 0.998, "R² Test": 0.887, "RMSE Train": 0.18, "RMSE Test": 1.26, "MAE Test": 0.61},
    {"Model": "LightGBM",          "R² Train": 0.991, "R² Test": 0.881, "RMSE Train": 0.37, "RMSE Test": 1.29, "MAE Test": 0.64},
    {"Model": "Random Forest",     "R² Train": 0.994, "R² Test": 0.876, "RMSE Train": 0.28, "RMSE Test": 1.31, "MAE Test": 0.66},
    {"Model": "Gradient Boosting", "R² Train": 0.942, "R² Test": 0.854, "RMSE Train": 0.87, "RMSE Test": 1.42, "MAE Test": 0.73},
    {"Model": "Decision Tree",     "R² Train": 0.979, "R² Test": 0.831, "RMSE Train": 0.52, "RMSE Test": 1.54, "MAE Test": 0.82},
    {"Model": "SVR",               "R² Train": 0.748, "R² Test": 0.741, "RMSE Train": 1.82, "RMSE Test": 1.90, "MAE Test": 1.01},
    {"Model": "Linear Regression", "R² Train": 0.612, "R² Test": 0.601, "RMSE Train": 2.26, "RMSE Test": 2.35, "MAE Test": 1.48},
])

FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": ["RAIN_LAG1","RAIN_ROLLING3","RAIN_LAG2","RAIN_ROLLING7",
                 "RAIN_INTENSITY","RAIN_LAG3","RAIN_CHANGE","RH2M",
                 "GWETTOP","T2MDEW","TEMP_DEW_DIFF","T2M_MIN","WS2M"],
    "Importance": [0.301,0.198,0.142,0.108,0.072,0.059,0.041,0.029,
                   0.018,0.012,0.009,0.007,0.004],
})

# ── Sidebar Navigation ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌧️ Rainfall ML")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Exploratory Analysis",
         "⚙️ Feature Engineering",
         "🤖 Model Comparison",
         "🎯 Best Model Results",
         "🔮 Future Predictions",
         "🔍 Live Prediction"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    if df is not None:
        st.markdown(f"- Rows: `{len(df):,}`")
        st.markdown(f"- Years: `{int(df['YEAR'].min())} – {int(df['YEAR'].max())}`")
        st.markdown(f"- Features: `{len(df.columns)}`")
    else:
        st.markdown("*CSV not found in outputs/*")
    st.markdown("---")
    st.caption("Best Model: **XGBoost** | R²: 0.887")

# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-header">
      <h1>🌧️ Rainfall Prediction using Machine Learning</h1>
      <p>A complete ML pipeline — data cleaning → feature engineering → model training → future forecasting (2021–2025)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
      <div class="metric-card">
        <div class="label">Best Model</div>
        <div class="value">XGBoost</div>
        <div class="sub">Lowest RMSE on test set</div>
      </div>
      <div class="metric-card">
        <div class="label">R² Score (Test)</div>
        <div class="value">0.887</div>
        <div class="sub">88.7% variance explained</div>
      </div>
      <div class="metric-card">
        <div class="label">RMSE (Test)</div>
        <div class="value">1.26 mm</div>
        <div class="sub">Root Mean Squared Error</div>
      </div>
      <div class="metric-card">
        <div class="label">Models Trained</div>
        <div class="value">7</div>
        <div class="sub">Linear → Ensemble → SVR</div>
      </div>
      <div class="metric-card">
        <div class="label">Features Used</div>
        <div class="value">13</div>
        <div class="sub">Lag, rolling & weather</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">Pipeline Overview</div>', unsafe_allow_html=True)
        steps = [
            ("1. Data Loading", "NASA POWER dataset — daily weather variables including PRECTOTCORR (rainfall), T2M, RH2M, WS2M, GWETTOP, T2MDEW, T2M_MIN"),
            ("2. Data Cleaning", "Replace -999 sentinel values with NaN, linear interpolation, forward/backward fill, remove negative rainfall"),
            ("3. Feature Engineering", "Lag features (1–3 days), rolling means (3/7 days), rain intensity ratio, temperature-dew diff"),
            ("4. Train/Test Split", "Pre-2021 data for training (80/20 split), 2021–2025 reserved for future predictions"),
            ("5. Scaling + PCA", "StandardScaler on all features; PCA at 95% explained variance for SVR"),
            ("6. Model Training", "7 models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVR"),
            ("7. Evaluation", "RMSE, MAE, R² on both train and test sets — XGBoost wins"),
            ("8. Future Forecast", "XGBoost predicts 2021–2025 rainfall, saved to CSV"),
        ]
        for title, desc in steps:
            with st.expander(title):
                st.write(desc)

    with col2:
        st.markdown('<div class="section-title">Quick Output Gallery</div>', unsafe_allow_html=True)
        gallery_images = {
            "Rainfall Distribution": os.path.join(OUTPUTS, "rainfall_distribution.png"),
            "Model Comparison": os.path.join(OUTPUTS, "model_comparision(RMSE).png"),
            "Feature Importance": os.path.join(OUTPUTS, "feature_importance.png"),
        }
        sel = st.selectbox("Select output", list(gallery_images.keys()))
        img_path = gallery_images[sel]
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Run the pipeline first to generate output images.")

# ════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Rainfall Over Time", "📦 Monthly Distribution", "🌡️ Correlation", "📋 Dataset Preview"]
    )

    with tab1:
        st.markdown('<div class="section-title">Rainfall Over Time</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUTS, "rainfall_over_time.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        elif df is not None:
            fig = px.line(df.sample(min(5000, len(df))).sort_values("DATE"),
                          x="DATE", y="PRECTOTCORR",
                          title="Daily Rainfall (mm) Over Time",
                          labels={"PRECTOTCORR": "Rainfall (mm)", "DATE": "Date"},
                          color_discrete_sequence=["#01696f"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("outputs/rainfall_over_time.png not found. Run the pipeline first.")

        img2 = os.path.join(OUTPUTS, "rainfall_distribution.png")
        if os.path.exists(img2):
            st.markdown('<div class="section-title">Rainfall Distribution (Histogram)</div>', unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
        elif df is not None:
            fig2 = px.histogram(df, x="PRECTOTCORR", nbins=60, title="Rainfall Distribution",
                                color_discrete_sequence=["#01696f"])
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Monthly Rainfall Distribution</div>', unsafe_allow_html=True)
        img_m = os.path.join(OUTPUTS, "montly_rainfall_distribution.png")
        img_avg = os.path.join(OUTPUTS, "Average_montly_rainfall.png")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(img_m):
                st.image(img_m, caption="Monthly Boxplot", use_container_width=True)
            elif df is not None:
                fig_box = px.box(df, x="MONTH", y="PRECTOTCORR",
                                 title="Monthly Rainfall Boxplot",
                                 labels={"MONTH": "Month", "PRECTOTCORR": "Rainfall (mm)"},
                                 color_discrete_sequence=["#01696f"])
                fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            if os.path.exists(img_avg):
                st.image(img_avg, caption="Average Monthly Rainfall", use_container_width=True)
            elif df is not None:
                month_avg = df.groupby("MONTH")["PRECTOTCORR"].mean().reset_index()
                month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                month_avg["Month"] = month_avg["MONTH"].map(month_names)
                fig_bar = px.bar(month_avg, x="Month", y="PRECTOTCORR",
                                 title="Average Monthly Rainfall",
                                 color="PRECTOTCORR",
                                 color_continuous_scale=["#cedcd8","#01696f","#0f3638"])
                fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        img_corr = os.path.join(OUTPUTS, "correlation_heatmap.png")
        if os.path.exists(img_corr):
            st.image(img_corr, use_container_width=True)
        elif df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            corr_mat = df[num_cols].corr()
            fig_corr = px.imshow(corr_mat, color_continuous_scale="RdBu_r",
                                 title="Correlation Heatmap", aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Run the pipeline to generate the correlation heatmap.")

    with tab4:
        st.markdown('<div class="section-title">Cleaned Dataset Preview</div>', unsafe_allow_html=True)
        if df is not None:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Rows", f"{len(df):,}")
            col_b.metric("Total Columns", len(df.columns))
            col_c.metric("Date Range", f"{df['DATE'].min().year} – {df['DATE'].max().year}")
            st.dataframe(df.head(100), use_container_width=True, height=400)
            st.download_button(
                "⬇️ Download Cleaned Dataset",
                df.to_csv(index=False).encode(),
                file_name="cleaned_rainfall_dataset.csv",
                mime="text/csv"
            )
        else:
            st.warning("outputs/cleaned_rainfall_dataset.csv not found. Run the pipeline first.")

# ════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")

    st.info("13 features were engineered from raw weather variables to improve predictive accuracy.")

    tab1, tab2 = st.tabs(["📌 Feature Descriptions", "📉 PCA Analysis"])

    with tab1:
        feature_table = pd.DataFrame([
            {"Feature": "RH2M",          "Type": "Raw",         "Description": "Relative humidity at 2m height (%)"},
            {"Feature": "GWETTOP",       "Type": "Raw",         "Description": "Surface soil wetness (top layer)"},
            {"Feature": "T2MDEW",        "Type": "Raw",         "Description": "Dew point temperature at 2m (°C)"},
            {"Feature": "T2M_MIN",       "Type": "Raw",         "Description": "Minimum air temperature at 2m (°C)"},
            {"Feature": "WS2M",          "Type": "Raw",         "Description": "Wind speed at 2m height (m/s)"},
            {"Feature": "RAIN_LAG1",     "Type": "Lag",         "Description": "Rainfall 1 day prior (mm)"},
            {"Feature": "RAIN_LAG2",     "Type": "Lag",         "Description": "Rainfall 2 days prior (mm)"},
            {"Feature": "RAIN_LAG3",     "Type": "Lag",         "Description": "Rainfall 3 days prior (mm)"},
            {"Feature": "RAIN_ROLLING3", "Type": "Rolling",     "Description": "3-day rolling mean rainfall (mm)"},
            {"Feature": "RAIN_ROLLING7", "Type": "Rolling",     "Description": "7-day rolling mean rainfall (mm)"},
            {"Feature": "RAIN_CHANGE",   "Type": "Derived",     "Description": "Day-over-day rainfall change (mm)"},
            {"Feature": "RAIN_INTENSITY","Type": "Derived",     "Description": "Today's rain / (7-day mean + 1) — burst intensity ratio"},
            {"Feature": "TEMP_DEW_DIFF", "Type": "Interaction", "Description": "T2M minus T2MDEW — atmospheric moisture deficit (°C)"},
        ])
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        img_fi = os.path.join(OUTPUTS, "feature_importance.png")
        if os.path.exists(img_fi):
            st.image(img_fi, use_container_width=True)
        else:
            fig_imp = px.bar(FEATURE_IMPORTANCE.sort_values("Importance"),
                             x="Importance", y="Feature", orientation="h",
                             title="XGBoost Feature Importance",
                             color="Importance",
                             color_continuous_scale=["#cedcd8","#01696f","#0f3638"])
            fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=420)
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">PCA — Explained Variance</div>', unsafe_allow_html=True)
        img_pca = os.path.join(OUTPUTS, "pca_variance.png")
        if os.path.exists(img_pca):
            st.image(img_pca, use_container_width=True)
        else:
            st.info("Run the pipeline to generate PCA variance plot.")
        st.markdown("""
        PCA was applied at **95% explained variance threshold**, reducing 13 features to fewer
        principal components for the SVR model only. All tree-based models (XGBoost, RF, LightGBM)
        used the original 13 features directly.
        """)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    tab1, tab2, tab3 = st.tabs(["📊 Metrics Table", "📉 RMSE Chart", "🏅 R² Comparison"])

    with tab1:
        st.markdown('<div class="section-title">All Models — Train / Test Metrics</div>', unsafe_allow_html=True)
        display_df = MODEL_RESULTS.copy()
        display_df["R² Train"] = display_df["R² Train"].map("{:.3f}".format)
        display_df["R² Test"]  = display_df["R² Test"].map("{:.3f}".format)
        display_df["RMSE Train"] = display_df["RMSE Train"].map("{:.3f}".format)
        display_df["RMSE Test"]  = display_df["RMSE Test"].map("{:.3f}".format)
        display_df["MAE Test"]   = display_df["MAE Test"].map("{:.3f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        img_mc = os.path.join(OUTPUTS, "model_comparision(RMSE).png")
        img_em = os.path.join(OUTPUTS, "evaluation_metrics.png")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(img_mc):
                st.image(img_mc, caption="RMSE Model Comparison", use_container_width=True)
        with c2:
            if os.path.exists(img_em):
                st.image(img_em, caption="Evaluation Metrics Overview", use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">RMSE on Test Set — All Models</div>', unsafe_allow_html=True)
        sorted_rmse = MODEL_RESULTS.sort_values("RMSE Test", ascending=True)
        fig_rmse = px.bar(
            sorted_rmse, x="RMSE Test", y="Model", orientation="h",
            title="Test RMSE (Lower is Better)",
            color="RMSE Test",
            color_continuous_scale=["#01696f","#cedcd8","#e9e0c6"],
            text="RMSE Test",
        )
        fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_rmse.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">R² Score — Train vs Test</div>', unsafe_allow_html=True)
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            name="R² Train", x=MODEL_RESULTS["Model"], y=MODEL_RESULTS["R² Train"],
            marker_color="#cedcd8"
        ))
        fig_r2.add_trace(go.Bar(
            name="R² Test", x=MODEL_RESULTS["Model"], y=MODEL_RESULTS["R² Test"],
            marker_color="#01696f"
        ))
        fig_r2.update_layout(
            barmode="group", title="R² Score — Train vs Test",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0, 1.05], title="R² Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        st.markdown("""
        > **Overfitting note:** XGBoost shows R² = 0.998 on train and 0.887 on test — some overfitting, 
        but test performance is still significantly best. LightGBM is a close second and is faster to train.
        """)

# ════════════════════════════════════════════════════════════════
# PAGE 5 — BEST MODEL RESULTS
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Best Model Results":
    st.title("🎯 Best Model — XGBoost Results")

    st.success("XGBoost achieved the lowest RMSE (1.26 mm) and highest R² (0.887) on the test set.")

    tab1, tab2, tab3 = st.tabs(
        ["📉 Actual vs Predicted", "📊 Residuals", "📈 Best Metrics"]
    )

    with tab1:
        st.markdown('<div class="section-title">Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            img_avp_test = os.path.join(OUTPUTS, "act_vs_pred(test).png")
            img_avp2     = os.path.join(OUTPUTS, "actual_vs_predicted.png")
            if os.path.exists(img_avp_test):
                st.image(img_avp_test, caption="Actual vs Predicted — Test Set", use_container_width=True)
            elif os.path.exists(img_avp2):
                st.image(img_avp2, caption="Actual vs Predicted", use_container_width=True)
            else:
                st.info("Run the pipeline to generate this plot.")
        with c2:
            img_avp_fut = os.path.join(OUTPUTS, "act_vs_pred(2021-2025).png")
            if os.path.exists(img_avp_fut):
                st.image(img_avp_fut, caption="Actual vs Predicted — 2021–2025", use_container_width=True)
            else:
                st.info("Future prediction comparison will appear here.")

    with tab2:
        st.markdown('<div class="section-title">Residual Analysis</div>', unsafe_allow_html=True)
        img_res = os.path.join(OUTPUTS, "residual_plot.png")
        if os.path.exists(img_res):
            st.image(img_res, use_container_width=True)
        else:
            st.info("Run the pipeline to generate the residual plot.")
        st.markdown("""
        **Interpreting residuals:**  
        - Residuals should be randomly scattered around zero — no clear pattern  
        - Heteroscedasticity (fan shape) at high rainfall values is expected since heavy rainfall is rare and harder to predict  
        """)

    with tab3:
        st.markdown('<div class="section-title">XGBoost Evaluation Metrics</div>', unsafe_allow_html=True)
        img_bme = os.path.join(OUTPUTS, "best_model_evaluation_metrics.png")
        if os.path.exists(img_bme):
            st.image(img_bme, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² (Test)",    "0.887",  "+48.6% vs Linear Reg")
        col2.metric("RMSE (Test)",  "1.26 mm", "-46% vs Linear Reg")
        col3.metric("MAE (Test)",   "0.61 mm", "—")
        col4.metric("n_estimators", "500",     "learning_rate=0.05")

# ════════════════════════════════════════════════════════════════
# PAGE 6 — FUTURE PREDICTIONS
# ════════════════════════════════════════════════════════════════
elif page == "🔮 Future Predictions":
    st.title("🔮 Future Predictions — 2021–2025")

    if pred_df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Monthly Aggregates", "📋 Raw Data"])

        with tab1:
            st.markdown('<div class="section-title">Actual vs Predicted Rainfall (2021–2025)</div>', unsafe_allow_html=True)
            sample = pred_df.head(500) if len(pred_df) > 500 else pred_df
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=sample["Date"], y=sample["Actual Rainfall (mm)"],
                name="Actual", line=dict(color="#0f3638", width=1.5)
            ))
            fig_ts.add_trace(go.Scatter(
                x=sample["Date"], y=sample["Predicted Rainfall (mm)"],
                name="Predicted (XGBoost)", line=dict(color="#01696f", width=1.5, dash="dot")
            ))
            fig_ts.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Date", yaxis_title="Rainfall (mm)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=400
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with tab2:
            pred_df["Year"]  = pred_df["Date"].dt.year
            pred_df["Month"] = pred_df["Date"].dt.month
            monthly = pred_df.groupby(["Year","Month"])[
                ["Actual Rainfall (mm)","Predicted Rainfall (mm)"]
            ].sum().reset_index()

            year_sel = st.selectbox("Select Year", sorted(monthly["Year"].unique()))
            yr_data = monthly[monthly["Year"] == year_sel]

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(name="Actual",    x=yr_data["Month"], y=yr_data["Actual Rainfall (mm)"],    marker_color="#0f3638"))
            fig_m.add_trace(go.Bar(name="Predicted", x=yr_data["Month"], y=yr_data["Predicted Rainfall (mm)"], marker_color="#01696f"))
            fig_m.update_layout(
                barmode="group", title=f"Monthly Rainfall — {year_sel}",
                xaxis=dict(title="Month", tickmode="array",
                           tickvals=list(range(1,13)),
                           ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
                yaxis_title="Rainfall (mm)",
                plot_bgcolor="white", paper_bgcolor="white", height=380
            )
            st.plotly_chart(fig_m, use_container_width=True)

        with tab3:
            st.dataframe(pred_df[["Date","Actual Rainfall (mm)","Predicted Rainfall (mm)"]],
                         use_container_width=True, height=420)
            st.download_button(
                "⬇️ Download Predictions CSV",
                pred_df.to_csv(index=False).encode(),
                file_name="rainfall_predictions_2021_2025.csv",
                mime="text/csv"
            )
    else:
        st.warning("outputs/rainfall_predictions_2021_2025.csv not found. Run the pipeline first.")

# ════════════════════════════════════════════════════════════════
# PAGE 7 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Rainfall Prediction")
    st.info("Enter weather feature values to get a simulated rainfall prediction using XGBoost model logic.")

    st.markdown('<div class="section-title">Input Weather Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        rh2m     = st.slider("Relative Humidity (%)", 0.0, 100.0, 75.0, 0.5)
        gwettop  = st.slider("Soil Wetness (GWETTOP)", 0.0, 1.0, 0.6, 0.01)
        t2mdew   = st.slider("Dew Point Temp (°C)", -10.0, 30.0, 18.0, 0.1)
        t2m_min  = st.slider("Min Air Temp (°C)", -5.0, 35.0, 20.0, 0.1)
        ws2m     = st.slider("Wind Speed 2m (m/s)", 0.0, 15.0, 2.5, 0.1)

    with col2:
        rain_lag1 = st.number_input("Rainfall 1 day ago (mm)", 0.0, 100.0, 2.5, 0.1)
        rain_lag2 = st.number_input("Rainfall 2 days ago (mm)", 0.0, 100.0, 1.8, 0.1)
        rain_lag3 = st.number_input("Rainfall 3 days ago (mm)", 0.0, 100.0, 1.2, 0.1)

    with col3:
        rain_rolling3 = (rain_lag1 + rain_lag2 + rain_lag3) / 3
        rain_rolling7 = rain_rolling3 * 0.85
        rain_change   = rain_lag1 - rain_lag2
        rain_intensity = rain_lag1 / (rain_rolling7 + 1)
        temp_dew_diff  = t2m_min - t2mdew

        st.markdown("**Auto-computed features:**")
        st.metric("RAIN_ROLLING3", f"{rain_rolling3:.2f} mm")
        st.metric("RAIN_ROLLING7", f"{rain_rolling7:.2f} mm")
        st.metric("RAIN_CHANGE",   f"{rain_change:+.2f} mm")
        st.metric("RAIN_INTENSITY",f"{rain_intensity:.3f}")
        st.metric("TEMP_DEW_DIFF", f"{temp_dew_diff:.1f} °C")

    st.markdown("---")
    if st.button("🌧️ Predict Rainfall", type="primary"):

        if model is None:
            st.error("❌ XGBoost model could not be loaded.")
        else:

            # Exact feature order used during model training
            MODEL_FEATURES = [
                "RH2M",
                "GWETTOP",
                "T2MDEW",
                "T2M_MIN",
                "WS2M",
                "RAIN_LAG1",
                "RAIN_LAG2",
                "RAIN_LAG3",
                "RAIN_ROLLING3",
                "RAIN_ROLLING7",
                "RAIN_CHANGE",
                "RAIN_INTENSITY",
                "TEMP_DEW_DIFF"
            ]

            # Create input DataFrame in the exact order expected by XGBoost
            input_data = pd.DataFrame([[
                rh2m,
                gwettop,
                t2mdew,
                t2m_min,
                ws2m,
                rain_lag1,
                rain_lag2,
                rain_lag3,
                rain_rolling3,
                rain_rolling7,
                rain_change,
                rain_intensity,
                temp_dew_diff
            ]], columns=MODEL_FEATURES)

            try:
                # Actual trained XGBoost prediction
                prediction = model.predict(input_data)

                pred = max(0.0, float(prediction[0]))

                # Rainfall category
                if pred < 0.5:
                    label = "No Rain / Trace"
                elif pred < 2.5:
                    label = "Light Rain"
                elif pred < 7.5:
                    label = "Moderate Rain"
                elif pred < 15:
                    label = "Heavy Rain"
                else:
                    label = "Very Heavy Rain"

                st.markdown("---")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Predicted Rainfall",
                    f"{pred:.2f} mm"
                )

                c2.metric(
                    "Rain Category",
                    label
                )

                st.success(
                    f"🌧️ XGBoost Forecast: **{label}** — "
                    f"**{pred:.2f} mm** expected"
                )

                with st.expander("🔍 View Model Input Features"):
                    st.dataframe(
                        input_data,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")