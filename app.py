import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PriceRight — Laptop Price Estimator", layout="wide")

CATEGORICAL_COLS = ["Company", "TypeName", "Cpu brand", "Gpu brand", "os"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body {
    color-scheme: dark only;
}
[data-testid="stApp"], [data-testid="stMain"], [data-testid="stHeader"],
[data-testid="stSidebar"] {
    background-color: #12161B !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #3A424B;
}
[data-testid="stMainBlockContainer"] {
    color: #EAEFF2 !important;
}
[data-testid="stWidgetLabel"] p {
    color: #EAEFF2 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1B2027 !important;
}
[data-testid="stSelectbox"] input,
[data-testid="stSliderTickBar"] p,
[data-testid="stSliderThumbValue"] p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #EAEFF2 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #3A424B;
}

.pr-doc-header {
    border-bottom: 3px solid #EAEFF2;
    padding-bottom: 14px;
    margin-bottom: 20px;
}
.pr-doc-header .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #8B94A0;
    text-transform: uppercase;
}
.pr-doc-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    letter-spacing: -0.01em;
    color: #EAEFF2;
    margin: 4px 0 6px 0;
}
.pr-doc-header p {
    font-size: 0.92rem;
    color: #A7AFB9;
    margin: 0;
    max-width: 60ch;
}

.pr-field-group {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #EAEFF2;
    border-bottom: 1px solid #3A424B;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #3A424B !important;
    border-radius: 4px !important;
    background: #1B2027;
}

div.stButton > button {
    background: #F5A623;
    color: #12161B;
    border: none;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 11px 18px;
    width: 100%;
}
div.stButton > button:hover {
    background: #ffbb47;
    color: #12161B;
}

.pr-tag-wrap {
    display: flex;
    justify-content: center;
    margin: 18px 0 6px 0;
}
.pr-tag {
    position: relative;
    background: #F5A623;
    clip-path: polygon(0% 50%, 13% 4%, 100% 4%, 100% 96%, 13% 96%);
    padding: 22px 34px 22px 54px;
    min-width: 300px;
    text-align: center;
    filter: drop-shadow(0 6px 14px rgba(0,0,0,0.45));
}
.pr-tag::before {
    content: "";
    position: absolute;
    left: 26px;
    top: 50%;
    transform: translateY(-50%);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #12161B;
    border: 2px solid #12161B;
}
.pr-tag .tag-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #12161B;
    opacity: 0.75;
}
.pr-tag .tag-amount {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: #12161B;
    line-height: 1.15;
}
.pr-footnote {
    text-align: center;
    font-size: 0.78rem;
    color: #8B94A0;
    margin-top: 10px;
}
.pr-empty-state {
    background: #1B2027;
    border: 1px dashed #3A424B;
    border-radius: 6px;
    padding: 18px 20px;
    color: #A7AFB9;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pr-doc-header">
  <div class="eyebrow">Pricing tool</div>
  <h1>PriceRight</h1>
  <p>Set the specs in the sidebar, then click Estimate Price to get a
  data driven price for a new laptop listing.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_bundle = joblib.load("model.pkl")
    return model_bundle["model"], model_bundle["columns"]


try:
    model, model_columns = load_model()
except FileNotFoundError:
    st.error("Model file not found. model.pkl must be in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

COMPANIES = ["Apple", "Asus", "Dell", "HP", "Lenovo", "MSI", "Toshiba", "Acer", "Other"]
TYPES = ["Ultrabook", "Notebook", "Gaming", "Netbook", "Workstation", "2 in 1 Convertible"]
CPU_BRANDS = ["Intel Core i7", "Intel Core i5", "Intel Core i3", "Other Intel Processor", "AMD Processor"]
GPU_BRANDS = ["Intel", "Nvidia", "AMD", "Other"]
OS_OPTIONS = ["Windows", "Mac", "No OS", "Linux/Other"]
RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]

if "history" not in st.session_state:
    st.session_state.history = []


def build_row(company, typename, ram, inches, weight, os_choice, resolution,
              touchscreen, ips, cpu_brand, cpu_speed, gpu_brand, ssd, hdd, flash, hybrid):
    x_res, y_res = (int(v) for v in resolution.split()[0].split("x"))
    ppi = float(np.sqrt(x_res**2 + y_res**2) / inches)
    return pd.DataFrame([{
        "Inches": inches, "Ram": ram, "Weight": weight,
        "Touchscreen": int(touchscreen), "IPS": int(ips), "PPI": ppi,
        "Cpu speed (GHz)": cpu_speed, "HDD": hdd, "SSD": ssd,
        "Hybrid": hybrid, "Flash_Storage": flash,
        "Company": company, "TypeName": typename,
        "Cpu brand": cpu_brand, "Gpu brand": gpu_brand, "os": os_choice,
    }])


def predict(row_df):
    encoded = pd.get_dummies(row_df, columns=CATEGORICAL_COLS, drop_first=True)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    price = model.predict(encoded)[0]
    return max(float(price), 0.0)

with st.sidebar:
    st.markdown('<div class="pr-field-group">Basic specs</div>', unsafe_allow_html=True)
    company = st.selectbox("Brand", COMPANIES)
    typename = st.selectbox("Laptop type", TYPES)
    ram = st.select_slider("RAM (GB)", options=RAM_OPTIONS, value=8)
    inches = st.slider("Screen size (inches)", 10.0, 18.4, 15.6, 0.1)
    weight = st.slider("Weight (kg)", 0.6, 4.5, 2.0, 0.1)
    os_choice = st.selectbox("Operating System", OS_OPTIONS)

    st.markdown('<div class="pr-field-group">§ Display</div>', unsafe_allow_html=True)
    resolution = st.selectbox(
        "Resolution",
        ["1366x768", "1920x1080 (Full HD)", "2560x1600", "2880x1800", "3840x2160 (4K)", "3200x1800"],
    )
    touchscreen = st.checkbox("Touchscreen")
    ips = st.checkbox("IPS panel", value=True)

    st.markdown('<div class="pr-field-group">Performance</div>', unsafe_allow_html=True)
    cpu_brand = st.selectbox("CPU", CPU_BRANDS)
    cpu_speed = st.slider("CPU clock speed (GHz)", 0.9, 3.6, 2.5, 0.1)
    gpu_brand = st.selectbox("GPU", GPU_BRANDS)

    st.markdown('<div class="pr-field-group">Storage (GB)</div>', unsafe_allow_html=True)
    ssd = st.select_slider("SSD", options=[0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000], value=256)
    hdd = st.select_slider("HDD", options=[0, 500, 1000, 2000], value=0)
    flash = st.select_slider("Flash storage", options=[0, 16, 32, 64, 128, 256], value=0)
    hybrid = st.select_slider("Hybrid", options=[0, 500, 1000, 2000], value=0)

    st.write("")
    go = st.button("ESTIMATE PRICE")

tab_predict, tab_about = st.tabs(["Prediction", "About"])

with tab_predict:
    if inches <= 0:
        st.error("Screen size must be greater than 0 inches.")
        st.stop()

    if go:
        try:
            row = build_row(company, typename, ram, inches, weight, os_choice, resolution,
                             touchscreen, ips, cpu_brand, cpu_speed, gpu_brand, ssd, hdd, flash, hybrid)
            predicted_price = predict(row)
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")
            st.stop()

        st.markdown(f"""
        <div class="pr-tag-wrap">
          <div class="pr-tag">
            <div class="tag-label">Est. fair price</div>
            <div class="tag-amount">S$ {predicted_price:,.2f}</div>
          </div>
        </div>
        <div class="pr-footnote">Random Forest model, trained on 1,270+ real laptop listings
        use as a reference, not a guaranteed sale price.</div>
        """, unsafe_allow_html=True)

        with st.expander("See the specs used for this estimate"):
            st.dataframe(row.T.rename(columns={0: "Value"}).astype(str))

        st.session_state.history.append({
            "Brand": company, "Type": typename, "RAM (GB)": ram,
            "Screen (in)": inches, "SSD (GB)": ssd,
            "Predicted price (SGD)": round(predicted_price, 2),
        })

        st.markdown("#### How the price responds to RAM")
        st.caption("Every other spec fixed, this shows what the model would predict "
                   "if only the RAM changed its useful for sanity check that the model behaves normally.")
        sweep_prices = []
        for r in RAM_OPTIONS:
            sweep_row = build_row(company, typename, r, inches, weight, os_choice, resolution,
                                   touchscreen, ips, cpu_brand, cpu_speed, gpu_brand, ssd, hdd, flash, hybrid)
            sweep_prices.append(predict(sweep_row))

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor("#12161B")
        ax.set_facecolor("#12161B")
        ax.plot(RAM_OPTIONS, sweep_prices, marker="o", color="#F5A623", linewidth=2)
        ax.axvline(ram, color="#EAEFF2", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("RAM (GB)", color="#EAEFF2")
        ax.set_ylabel("Predicted price (SGD)", color="#EAEFF2")
        ax.tick_params(colors="#A7AFB9")
        for spine in ax.spines.values():
            spine.set_color("#3A424B")
        ax.grid(color="#3A424B", linewidth=0.5, alpha=0.6)
        st.pyplot(fig)
    else:
        st.markdown(
            '<div class="pr-empty-state">Set the specs in the sidebar, then click '
            '<b>Estimate Price</b>.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        st.markdown("#### Prediction history")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button(
            "Download history as CSV",
            data=hist_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

with tab_about:
    st.markdown("""
    **What this tool does**

    PriceRight helps an online laptop retailer set a fair, competitive price for a
    **new laptop model**, just from its specs instead of manually researching every
    competitor listing by manually.

    **Model**: Random Forest Regressor, chosen after comparing Linear Regression,
    Decision Tree, Random Forest and Gradient Boosting, then tuned with
    `RandomizedSearchCV`.

    **Data**: 1,270+ real laptop listings (brand, type, screen, CPU, RAM, storage,
    GPU, OS, weight and price in SGD).

    **A note on the RAM chart**: it's a *sensitivity check*, not a guarantee, it
    shows how the model's own prediction shifts as one spec changes, everything
    else is constant.
    """)
