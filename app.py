# ============================================================
# FLUOROSIS RISK CALCULATOR – Streamlit Web App
# Thal Desert Children, Pakistan
# Run: streamlit run app.py
# ============================================================

# ── Install dependencies (for Google Colab / first run) ──────
# !pip install streamlit xgboost shap openpyxl joblib scikit-learn

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fluorosis Risk Calculator",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fluorosis_model.pkl")

GRADE_LABELS = {
    0: "Normal",
    1: "Very Mild / Questionable",
    2: "Mild Fluorosis",
    3: "Moderate Fluorosis",
    4: "Moderately Severe Fluorosis",
    5: "Severe Fluorosis",
}

GENO_MAP = {"FF":0,"Ff":1,"ff":2,"BB":0,"Bb":1,"bb":2,
            "TT":0,"Tt":1,"tt":2,"AA":0,"Aa":1,"aa":2}

FEATURES = ["Age","Sex_enc","Weight_kg",
            "Water_Fluoride_mgL","Daily_Water_L",
            "FokI_enc","BsmI_enc","TaqI_enc","ApaI_enc",
            "Calcium_mgdL","Vit_D_ngmL",
            "TaqI_x_Fluoride","FokI_x_Fluoride",
            "BsmI_x_Fluoride","ApaI_x_Fluoride","Fluoride_x_DailyL"]

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️  Model file not found. Please run fluorosis_calculator.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

def build_feature_vector(age, sex, bmi, water_fluoride, daily_water,
                          calcium, vit_d, foki, bsmi, taqi, apai):
    sex_e  = 1 if sex == "M" else 0
    foki_e = GENO_MAP[foki]
    bsmi_e = GENO_MAP[bsmi]
    taqi_e = GENO_MAP[taqi]
    apai_e = GENO_MAP[apai]
    weight_est = bmi * ((age * 0.06 + 1.0) ** 2)
    return np.array([[
        age, sex_e, weight_est,
        water_fluoride, daily_water,
        foki_e, bsmi_e, taqi_e, apai_e,
        calcium, vit_d,
        taqi_e * water_fluoride,
        foki_e * water_fluoride,
        bsmi_e * water_fluoride,
        apai_e * water_fluoride,
        water_fluoride * daily_water,
    ]])

def predict_risk(feat_vec, meta):
    mdl   = meta["model"]
    proba = mdl.predict_proba(feat_vec)[0]
    grade = int(mdl.predict(feat_vec)[0])
    grades = mdl.classes_
    risk_score = sum(g * p for g, p in zip(grades, proba)) / 5.0 * 100
    return risk_score, grade, dict(zip(grades, proba.round(3)))

def get_shap_contributions(feat_vec, meta):
    import shap as shap_lib
    expl = meta["explainer"]
    sv   = expl.shap_values(feat_vec)
    arr  = np.abs(np.array(sv))
    feat_axes = [i for i, s in enumerate(arr.shape) if s == len(FEATURES)]
    feat_ax   = feat_axes[-1] if feat_axes else (arr.ndim - 1)
    other_axes = tuple(i for i in range(arr.ndim) if i != feat_ax)
    sv_abs = arr.mean(axis=other_axes) if other_axes else arr
    gene_feats     = [f for f in FEATURES if any(k in f for k in ["FokI","BsmI","TaqI","ApaI"])]
    fluoride_feats = [f for f in FEATURES if "Fluoride" in f or "Water" in f or "DailyL" in f]
    g_idx  = [FEATURES.index(f) for f in gene_feats]
    fl_idx = [FEATURES.index(f) for f in fluoride_feats]
    tot    = sv_abs.sum() if sv_abs.sum() > 0 else 1
    return (sv_abs[g_idx].sum() / tot * 100,
            sv_abs[fl_idx].sum() / tot * 100,
            sv_abs, gene_feats, fluoride_feats)

def risk_color(score):
    if score < 25:  return "#4CAF50"
    if score < 50:  return "#FFC107"
    if score < 75:  return "#FF9800"
    return "#F44336"

def risk_label(score):
    if score < 25:  return "🟢 LOW RISK"
    if score < 50:  return "🟡 MODERATE RISK"
    if score < 75:  return "🟠 HIGH RISK"
    return "🔴 VERY HIGH RISK"

def recommendations(water_fluoride, vit_d, calcium, taqi, foki, daily_water):
    recs = []
    if water_fluoride > 1.5:
        recs.append(f"Water fluoride ({water_fluoride} mg/L) exceeds WHO limit. Use defluoridation filter or safe water.")
    if vit_d < 20:
        recs.append(f"Vitamin D deficient ({vit_d} ng/mL). Supplement and sunlight exposure recommended.")
    if calcium < 8.5:
        recs.append(f"Low calcium ({calcium} mg/dL). Increase dairy intake.")
    if taqi == "tt":
        recs.append("VDR TaqI 'tt' genotype – highest genetic susceptibility to fluorosis.")
    if foki == "ff":
        recs.append("VDR FokI 'ff' genotype – reduced VDR function; calcium supplementation may help.")
    if daily_water > 3.0 and water_fluoride > 1.5:
        recs.append("High daily water intake combined with elevated fluoride = significant cumulative exposure.")
    if not recs:
        recs.append("No immediate high-risk factors. Continue regular dental check-ups.")
    return recs

def shap_bar_figure(sv_abs, gene_feats, fluoride_feats):
    feat_colors = []
    for f in FEATURES:
        if f in gene_feats:       feat_colors.append("#E53935")
        elif f in fluoride_feats: feat_colors.append("#1E88E5")
        else:                      feat_colors.append("#43A047")

    sorted_idx = np.argsort(sv_abs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([FEATURES[i] for i in sorted_idx],
            sv_abs[sorted_idx],
            color=[feat_colors[i] for i in sorted_idx],
            alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title("SHAP – This Child's Risk Drivers", fontsize=11, fontweight="bold")
    patches = [
        mpatches.Patch(color="#E53935", label="Gene factors"),
        mpatches.Patch(color="#1E88E5", label="Fluoride factors"),
        mpatches.Patch(color="#43A047", label="Other factors"),
    ]
    ax.legend(handles=patches, fontsize=8)
    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/"
                 "Flag_of_Pakistan.svg/120px-Flag_of_Pakistan.svg.png", width=80)
st.sidebar.title("🦷 Fluorosis Risk Calculator")
st.sidebar.caption("Thal Desert Children Study · Pakistan")
page = st.sidebar.radio("Navigation",
                         ["Single Child Assessment", "Bulk Upload", "Village Risk Map"])

meta = load_model()

# ─────────────────────────────────────────────────────────────
# PAGE 1: Single Child Assessment
# ─────────────────────────────────────────────────────────────
if page == "Single Child Assessment":
    st.title("🦷 Fluorosis Risk Calculator")
    st.markdown("**AI-powered dental fluorosis risk assessment** | Thal Desert, Pakistan")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Child Profile")
        child_id      = st.text_input("Child ID", value="TDC-001")
        age           = st.slider("Age (years)", 6, 15, 10)
        sex           = st.selectbox("Sex", ["M", "F"])
        bmi           = st.number_input("BMI (kg/m²)", 12.0, 35.0, 17.5, step=0.1)
        village       = st.selectbox("Village",
                            ["Darya Khan","Mankera","Bhakkar","Karor","Layyah","Other"])

    with col2:
        st.subheader("💧 Fluoride Exposure")
        water_fluoride = st.number_input("Water Fluoride (mg/L)", 0.1, 10.0, 2.5, step=0.1)
        daily_water    = st.number_input("Daily Water Intake (L)", 0.5, 5.0, 2.0, step=0.1)
        st.markdown("---")
        st.subheader("🧪 Blood Parameters")
        calcium = st.number_input("Calcium (mg/dL)", 6.0, 12.0, 9.5, step=0.1)
        vit_d   = st.number_input("Vitamin D (ng/mL)", 5.0, 80.0, 22.0, step=0.5)

    with col3:
        st.subheader("🧬 VDR Genotypes")
        foki = st.selectbox("VDR FokI", ["FF", "Ff", "ff"])
        bsmi = st.selectbox("VDR BsmI", ["BB", "Bb", "bb"])
        taqi = st.selectbox("VDR TaqI", ["TT", "Tt", "tt"])
        apai = st.selectbox("VDR ApaI", ["AA", "Aa", "aa"])

    st.markdown("---")
    calc_btn = st.button("🔍 Calculate Fluorosis Risk", type="primary", use_container_width=True)

    if calc_btn:
        feat_vec = build_feature_vector(age, sex, bmi, water_fluoride, daily_water,
                                         calcium, vit_d, foki, bsmi, taqi, apai)
        risk_score, grade, proba_dict = predict_risk(feat_vec, meta)
        g_pct, fl_pct, sv_abs, gene_feats, fluoride_feats = \
            get_shap_contributions(feat_vec, meta)
        recs = recommendations(water_fluoride, vit_d, calcium, taqi, foki, daily_water)
        color = risk_color(risk_score)
        label = risk_label(risk_score)

        st.markdown("## 📊 Assessment Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Score", f"{risk_score:.1f}%")
        r2.metric("Dean's Grade", f"{grade} – {GRADE_LABELS[grade]}")
        r3.metric("Gene Contribution", f"{g_pct:.1f}%")
        r4.metric("Fluoride Contribution", f"{fl_pct:.1f}%")

        # Colour banner
        st.markdown(
            f"<div style='background:{color};padding:16px 24px;border-radius:10px;"
            f"color:white;font-size:1.5rem;font-weight:bold;text-align:center'>"
            f"{label}</div>",
            unsafe_allow_html=True
        )

        # Progress bar
        st.markdown(f"**Risk Score: {risk_score:.1f}%**")
        st.progress(int(risk_score))

        # Grade probability table
        st.markdown("### 📈 Grade Probability Distribution")
        prob_df = pd.DataFrame({"Grade": list(proba_dict.keys()),
                                 "Probability": list(proba_dict.values())})
        st.bar_chart(prob_df.set_index("Grade"))

        # SHAP explanation
        st.markdown("### 🔬 SHAP Explanation – Risk Drivers for This Child")
        shap_fig = shap_bar_figure(sv_abs, gene_feats, fluoride_feats)
        st.pyplot(shap_fig)
        plt.close()

        # Recommendations
        st.markdown("### 📋 Personalised Action Plan")
        for rec in recs:
            if "⚠" in rec or "exceeds" in rec.lower():
                st.warning(rec)
            else:
                st.success(rec)

        # Print-friendly report text
        st.markdown("### 🖨️ Print Report")
        report_lines = [
            f"FLUOROSIS RISK REPORT | Child: {child_id} | Village: {village}",
            f"{'─'*55}",
            f"Age: {age} yrs | Sex: {sex} | BMI: {bmi} kg/m²",
            f"Water Fluoride: {water_fluoride} mg/L | Daily Water: {daily_water} L",
            f"Calcium: {calcium} mg/dL | Vitamin D: {vit_d} ng/mL",
            f"VDR FokI: {foki} | BsmI: {bsmi} | TaqI: {taqi} | ApaI: {apai}",
            f"{'─'*55}",
            f"Risk Score   : {risk_score:.1f}/100",
            f"Risk Category: {label}",
            f"Predicted Dean's Grade: {grade} – {GRADE_LABELS[grade]}",
            f"Gene Contribution: {g_pct:.1f}% | Fluoride Contribution: {fl_pct:.1f}%",
            f"{'─'*55}",
            "ACTION PLAN:",
        ] + [f"  • {r}" for r in recs]
        report_text = "\n".join(report_lines)
        st.text_area("Report (copy or print)", report_text, height=260)
        st.download_button("⬇️ Download Report (.txt)",
                           data=report_text.encode(),
                           file_name=f"fluorosis_report_{child_id}.txt",
                           mime="text/plain")

# ─────────────────────────────────────────────────────────────
# PAGE 2: Bulk Upload
# ─────────────────────────────────────────────────────────────
elif page == "Bulk Upload":
    st.title("📂 Bulk Upload – Multiple Children")
    st.markdown("Upload an Excel file with the same columns as `data.xlsx`.")

    template_cols = ["Child_ID","Age","Sex","Weight_kg","Village",
                     "Water_Fluoride_mgL","Daily_Water_L",
                     "VDR_FokI","VDR_BsmI","VDR_TaqI","VDR_ApaI",
                     "Calcium_mgdL","Vit_D_ngmL"]
    st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in template_cols))

    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

    if uploaded:
        df_up = pd.read_excel(uploaded)
        st.write(f"Loaded {len(df_up)} rows.")

        # Check columns
        missing = [c for c in template_cols if c not in df_up.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            results = []
            progress = st.progress(0)
            for i, row in df_up.iterrows():
                try:
                    bmi_est = row["Weight_kg"] / ((row["Age"] * 0.06 + 1.0) ** 2)
                    fv = build_feature_vector(
                        int(row["Age"]), str(row["Sex"]), float(bmi_est),
                        float(row["Water_Fluoride_mgL"]), float(row["Daily_Water_L"]),
                        float(row["Calcium_mgdL"]), float(row.get("Vit_D_ngmL", 20)),
                        str(row["VDR_FokI"]), str(row["VDR_BsmI"]),
                        str(row["VDR_TaqI"]), str(row["VDR_ApaI"])
                    )
                    rs, gr, _ = predict_risk(fv, meta)
                    results.append({
                        "Child_ID": row["Child_ID"],
                        "Village":  row.get("Village","—"),
                        "Risk_Score": round(rs, 1),
                        "Predicted_Grade": gr,
                        "Grade_Label": GRADE_LABELS[gr],
                        "Risk_Category": risk_label(rs),
                    })
                except Exception as e:
                    results.append({"Child_ID": row.get("Child_ID","?"),
                                    "Error": str(e)})
                progress.progress((i + 1) / len(df_up))

            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)

            # Download results
            buf = io.BytesIO()
            res_df.to_excel(buf, index=False)
            st.download_button("⬇️ Download Results (.xlsx)",
                               data=buf.getvalue(),
                               file_name="fluorosis_bulk_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Grade distribution bar chart
            st.markdown("### Grade Distribution in Uploaded Data")
            grade_counts = res_df["Predicted_Grade"].value_counts().sort_index()
            st.bar_chart(grade_counts)

# ─────────────────────────────────────────────────────────────
# PAGE 3: Village Risk Map Table
# ─────────────────────────────────────────────────────────────
elif page == "Village Risk Map":
    st.title("🗺️ Village-wise Risk Summary")

    base = os.path.dirname(__file__)
    data_path = (os.path.join(base, "data.xlsx")
                 if os.path.exists(os.path.join(base, "data.xlsx"))
                 else os.path.join(base, "data_new.xlsx"))
    if not os.path.exists(data_path):
        st.error("data.xlsx / data_new.xlsx not found. Please run fluorosis_calculator.py first.")
    else:
        df_v = pd.read_excel(data_path, sheet_name="Village_Summary")
        st.dataframe(df_v, use_container_width=True)

        # Bar chart
        st.markdown("### Mean Fluoride by Village")
        chart_df = df_v.set_index("Village")[["Mean_Fluoride"]].sort_values("Mean_Fluoride", ascending=False)
        st.bar_chart(chart_df)

        st.markdown("### % Children with Fluorosis Grade ≥2")
        chart_df2 = df_v.set_index("Village")[["Pct_Grade_2plus"]].sort_values("Pct_Grade_2plus", ascending=False)
        st.bar_chart(chart_df2)

        # WHO threshold flag
        st.markdown("### ⚠️  Villages Exceeding WHO Fluoride Limit (1.5 mg/L)")
        exceed = df_v[df_v["Mean_Fluoride"] > 1.5][["Village","Mean_Fluoride","Mean_Grade","Pct_Grade_2plus"]]
        if len(exceed):
            st.error("The following villages have mean fluoride > 1.5 mg/L:")
            st.dataframe(exceed, use_container_width=True)
        else:
            st.success("No villages exceed WHO fluoride limit on average.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️  Replace `data.xlsx` with your real data to get real results. "
    "This tool is for research purposes only and does not substitute clinical judgment."
)
st.caption("Developed for: Dental Fluorosis Study – Thal Desert, Pakistan | "
           "VDR Gene Polymorphisms & Water Fluoride")
