"""
=============================================================
  STEP 3: STREAMLIT WEB APP — PROPERTY LEAD SCORING
=============================================================
  This is the front-end dashboard. A salesperson fills in
  a new lead's details, clicks Predict, and sees:
    ✅ A Green / Yellow / Red result
    📊 Why the model made that decision
    💡 A recommended next action for the sales team

  To run:  streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================
# PAGE CONFIGURATION  (must be the very first Streamlit call)
# =============================================================
st.set_page_config(
    page_title="Property Lead Scorer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================
# CUSTOM CSS  — make it look professional
# =============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Green result card */
    .result-green {
        background: linear-gradient(135deg, #1a7a4a, #2ecc71);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(46,204,113,0.4);
    }
    /* Yellow result card */
    .result-yellow {
        background: linear-gradient(135deg, #b8860b, #f39c12);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(243,156,18,0.4);
    }
    /* Red result card */
    .result-red {
        background: linear-gradient(135deg, #a93226, #e74c3c);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(231,76,60,0.4);
    }
    /* Action box */
    .action-box {
        background: #ffffff;
        border-left: 5px solid #3498db;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1rem;
    }
    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# LOAD MODEL AND ENCODERS
# =============================================================
@st.cache_resource   # Cache so we don't reload on every interaction
def load_model():
    """Load the trained model and supporting files from disk."""
    required = [
        "lead_scoring_model.pkl",
        "label_encoders.pkl",
        "feature_names.pkl",
        "feature_importance.pkl"
    ]
    for f in required:
        if not os.path.exists(f):
            return None, None, None, None

    model      = joblib.load("lead_scoring_model.pkl")
    encoders   = joblib.load("label_encoders.pkl")
    feat_names = joblib.load("feature_names.pkl")
    feat_imp   = joblib.load("feature_importance.pkl")
    return model, encoders, feat_names, feat_imp


model, encoders, feat_names, feat_imp_df = load_model()


# =============================================================
# SIDEBAR — Branding & Instructions
# =============================================================
with st.sidebar:
    st.image("https://via.placeholder.com/250x80/1a7a4a/ffffff?text=Property+AI", use_column_width=True)
    st.markdown("## 🏠 Lead Scoring AI")
    st.markdown("""
    **How to use this tool:**
    1. Fill in the buyer's details on the right
    2. Click **Predict Lead Quality**
    3. See the result and recommended action

    ---
    **What the colours mean:**
    - 🟢 **Green** — High-value, act immediately
    - 🟡 **Yellow** — Potential, needs nurturing
    - 🔴 **Red** — Low priority, automate follow-up
    """)

    # Show confusion matrix if it exists
    if os.path.exists("confusion_matrix.png"):
        st.markdown("---")
        st.markdown("**Model Accuracy:**")
        st.image("confusion_matrix.png", use_column_width=True)


# =============================================================
# MAIN PAGE HEADER
# =============================================================
st.title("🏠 Property Lead Quality Predictor")
st.markdown("*Powered by Random Forest AI — helps your sales team focus on the right buyers*")
st.markdown("---")


# =============================================================
# CHECK IF MODEL IS LOADED
# =============================================================
if model is None:
    st.error("""
    ⚠️ **Model not found!**

    Please run these two scripts first in your terminal:
    ```
    python generate_data.py
    python train_model.py
    ```
    Then refresh this page.
    """)
    st.stop()   # halt — nothing else to show


# =============================================================
# INPUT FORM — Salesperson enters the lead's details
# =============================================================
st.subheader("📋 Enter Lead Details")

# We use columns to create a neat multi-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**💰 Financial Profile**")

    monthly_income = st.number_input(
        "Monthly Income (RM)",
        min_value=500,
        max_value=50000,
        value=5000,
        step=500,
        help="Gross monthly salary or business income"
    )

    employment_type = st.selectbox(
        "Employment Type",
        options=["Government", "Private", "Self-Employed", "Unemployed"],
        help="Type of employment affects loan eligibility"
    )

    existing_loan = st.number_input(
        "Existing Monthly Loan Commitment (RM)",
        min_value=0,
        max_value=20000,
        value=800,
        step=100,
        help="Car loan + personal loan + any other commitments"
    )

    ccris_issue = st.radio(
        "CCRIS / CTOS Issues?",
        options=["No", "Yes"],
        help="Any known bad credit history?"
    )


with col2:
    st.markdown("**🏠 Project Fit**")

    property_price = st.selectbox(
        "Property Price (RM)",
        options=[350000, 450000, 550000, 650000, 800000, 1000000],
        format_func=lambda x: f"RM {x:,}",
        help="The specific unit the lead is interested in"
    )

    budget_stated = st.number_input(
        "Lead's Stated Budget (RM)",
        min_value=100000,
        max_value=2000000,
        value=500000,
        step=10000,
        format="%d",
        help="What budget did the lead mention?"
    )

    location_match = st.selectbox(
        "Location Match",
        options=["Perfect", "Acceptable", "Not Ideal"],
        help="Does the project location suit the lead's preference?"
    )

    bedroom_match = st.selectbox(
        "Bedroom Requirement Match",
        options=["Match", "Partial", "No Match"],
        help="Does the unit match their bedroom needs?"
    )


with col3:
    st.markdown("**📱 Engagement & Intent**")

    lead_source = st.selectbox(
        "Lead Source",
        options=["Facebook Ad", "Walk-in", "Agent Referral", "Website", "Billboard"],
        help="How did the lead find out about this project?"
    )

    response_count = st.slider(
        "Number of Responses / Replies",
        min_value=0,
        max_value=20,
        value=3,
        help="How many times has the lead replied to messages or calls?"
    )

    days_since_contact = st.slider(
        "Days Since First Contact",
        min_value=0,
        max_value=90,
        value=7,
        help="How long ago did you first engage this lead?"
    )

    showroom_visit = st.radio(
        "Has Visited Showroom?",
        options=["No", "Yes"],
        help="Physical showroom visits are a strong buying signal"
    )

    buy_purpose = st.selectbox(
        "Purpose of Purchase",
        options=["Own Stay", "Investment", "Both"],
    )

    buy_timeline_months = st.selectbox(
        "Buying Timeline",
        options=[1, 3, 6, 12, 24],
        index=2,
        format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
        help="How soon does the lead plan to buy?"
    )


st.markdown("---")


# =============================================================
# PREDICT BUTTON
# =============================================================
predict_btn = st.button("🔍 Predict Lead Quality", type="primary", use_container_width=True)

if predict_btn:

    # ── Compute derived features (same as training data) ──
    dti_ratio    = round(existing_loan / monthly_income, 3) if monthly_income > 0 else 1.0
    budget_fit   = round(budget_stated / property_price, 3) if property_price > 0 else 0.0

    # ── Build the input row as a dictionary ──
    input_dict = {
        "monthly_income_rm":    monthly_income,
        "employment_type":      employment_type,
        "existing_loan_rm":     existing_loan,
        "debt_to_income_ratio": dti_ratio,
        "ccris_issue":          1 if ccris_issue == "Yes" else 0,
        "property_price_rm":    property_price,
        "budget_stated_rm":     budget_stated,
        "budget_fit_ratio":     budget_fit,
        "location_match":       location_match,
        "bedroom_match":        bedroom_match,
        "lead_source":          lead_source,
        "response_count":       response_count,
        "days_since_contact":   days_since_contact,
        "showroom_visit":       1 if showroom_visit == "Yes" else 0,
        "buy_purpose":          buy_purpose,
        "buy_timeline_months":  buy_timeline_months,
    }

    # ── Convert to DataFrame ──
    input_df = pd.DataFrame([input_dict])

    # ── Encode text columns using saved encoders ──
    text_cols = ["employment_type", "location_match", "bedroom_match",
                 "lead_source", "buy_purpose"]

    for col in text_cols:
        le = encoders[col]
        # Handle unseen labels gracefully
        val = input_df[col].iloc[0]
        if val in le.classes_:
            input_df[col] = le.transform([val])
        else:
            input_df[col] = 0  # fallback

    # ── Ensure column order matches training ──
    input_df = input_df[feat_names]

    # ── Make prediction ──
    pred_label  = model.predict(input_df)[0]          # 0, 1, or 2
    pred_proba  = model.predict_proba(input_df)[0]    # [P(Red), P(Yellow), P(Green)]

    # ==========================================================
    # DISPLAY RESULT
    # ==========================================================
    st.markdown("## 🎯 Prediction Result")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        # ── Colour-coded result card ──
        if pred_label == 2:
            st.markdown(
                '<div class="result-green">🟢 GREEN LEAD<br>'
                '<span style="font-size:1rem">High Priority — Act Now!</span></div>',
                unsafe_allow_html=True
            )
        elif pred_label == 1:
            st.markdown(
                '<div class="result-yellow">🟡 YELLOW LEAD<br>'
                '<span style="font-size:1rem">Potential — Needs Nurturing</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-red">🔴 RED LEAD<br>'
                '<span style="font-size:1rem">Low Priority — Automate</span></div>',
                unsafe_allow_html=True
            )

        # ── Confidence bar chart ──
        st.markdown("**Model Confidence:**")
        conf_df = pd.DataFrame({
            "Category":    ["🔴 Red", "🟡 Yellow", "🟢 Green"],
            "Probability": [p * 100 for p in pred_proba]
        })

        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.barh(conf_df["Category"], conf_df["Probability"], color=colors, edgecolor="white")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)", fontsize=10)
        ax.set_title("Prediction Confidence", fontsize=11)
        for bar, val in zip(bars, conf_df["Probability"]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with res_col2:
        # ── Recommended Next Action ──
        st.markdown("### 💡 Recommended Next Action")

        if pred_label == 2:
            st.markdown("""
            <div class="action-box">
            🚀 <strong>Assign to Top Performer immediately</strong><br><br>
            • Send a personalised showroom invitation via WhatsApp today<br>
            • Prepare a booking package with early-bird pricing<br>
            • Schedule a follow-up call within <strong>24 hours</strong><br>
            • Priority: <span style="color:#2ecc71;font-weight:bold">HIGHEST</span>
            </div>
            """, unsafe_allow_html=True)

        elif pred_label == 1:
            # Personalise advice based on which blocker is detected
            blockers = []
            if dti_ratio > 0.50:
                blockers.append("⚠️ High debt ratio — suggest a lower-priced unit or joint applicant option (e.g., buy with spouse)")
            if budget_fit < 0.80:
                blockers.append("⚠️ Budget gap — show a unit in the RM {:,}–{:,} range instead".format(
                    int(budget_stated * 0.9), int(budget_stated * 1.1)))
            if response_count < 3:
                blockers.append("⚠️ Low engagement — try a different channel (e.g., WhatsApp voice note instead of text)")
            if buy_timeline_months >= 12:
                blockers.append("⚠️ Long timeline — share a limited-time offer to create urgency")
            if not blockers:
                blockers.append("ℹ️ Minor concerns — keep warm with monthly project updates")

            blocker_html = "".join(f"<li>{b}</li>" for b in blockers)
            st.markdown(f"""
            <div class="action-box">
            🔧 <strong>Specific blockers to address:</strong>
            <ul>{blocker_html}</ul>
            📅 Re-assess this lead in <strong>2–4 weeks</strong> after follow-up
            </div>
            """, unsafe_allow_html=True)

        else:
            red_reasons = []
            if ccris_issue == "Yes":
                red_reasons.append("❌ CCRIS/CTOS issue — loan approval very unlikely right now")
            if dti_ratio > 0.65:
                red_reasons.append("❌ Debt-to-income ratio too high for bank financing")
            if monthly_income < 2500:
                red_reasons.append("❌ Income likely insufficient for this property price range")
            if response_count == 0:
                red_reasons.append("❌ Zero responses — lead may be uncontactable")
            if employment_type == "Unemployed":
                red_reasons.append("❌ Currently unemployed — cannot obtain bank loan")
            if not red_reasons:
                red_reasons.append("ℹ️ Multiple weak signals across financial and engagement factors")

            reason_html = "".join(f"<li>{r}</li>" for r in red_reasons)
            st.markdown(f"""
            <div class="action-box">
            ⏳ <strong>Move to automated nurturing sequence</strong><br><br>
            <ul>{reason_html}</ul>
            📧 Add to <strong>monthly email newsletter</strong> only<br>
            🚫 <strong>Do not assign</strong> sales team calling time
            </div>
            """, unsafe_allow_html=True)

        # ── Key computed metrics ──
        st.markdown("### 📐 Computed Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Debt-to-Income Ratio", f"{dti_ratio:.2f}",
                  delta="✅ Good" if dti_ratio < 0.50 else "⚠️ High",
                  delta_color="off")
        m2.metric("Budget Fit Ratio", f"{budget_fit:.2f}",
                  delta="✅ Match" if budget_fit >= 0.90 else "⚠️ Gap",
                  delta_color="off")


    # ==========================================================
    # FEATURE IMPORTANCE BREAKDOWN
    # ==========================================================
    st.markdown("---")
    st.markdown("### 🔍 Why Did the Model Decide This?")
    st.markdown("*The chart below shows which features matter most to the model overall.*")

    top10 = feat_imp_df.head(10).sort_values("Importance")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bar_colors = ["#3498db"] * 10
    ax2.barh(top10["Feature"], top10["Importance"] * 100,
             color=bar_colors, edgecolor="white")
    ax2.set_xlabel("Importance (%)", fontsize=11)
    ax2.set_title("Top 10 Most Influential Features", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# =============================================================
# BATCH UPLOAD SECTION  (bonus feature)
# =============================================================
st.markdown("---")
st.subheader("📁 Batch Predict — Upload a CSV File")
st.markdown("Have a whole list of leads? Upload a CSV with the same columns and score them all at once.")

uploaded_file = st.file_uploader("Upload leads CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Encode text columns
        for col in ["employment_type", "location_match", "bedroom_match",
                    "lead_source", "buy_purpose"]:
            if col in batch_df.columns:
                le = encoders[col]
                batch_df[col] = batch_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Drop label column if present
        X_batch = batch_df[[c for c in feat_names if c in batch_df.columns]]

        preds   = model.predict(X_batch)
        probas  = model.predict_proba(X_batch)

        label_map = {0: "🔴 Red", 1: "🟡 Yellow", 2: "🟢 Green"}
        batch_df["Prediction"]        = [label_map[p] for p in preds]
        batch_df["Green_Probability"] = (probas[:, 2] * 100).round(1)

        # Sort: Greens first, then Yellows, then Reds
        batch_df["_sort"] = preds
        batch_df = batch_df.sort_values("_sort", ascending=False).drop("_sort", axis=1)

        st.success(f"✅ Scored {len(batch_df):,} leads!")
        st.dataframe(batch_df[["Prediction", "Green_Probability"] +
                               [c for c in batch_df.columns
                                if c not in ["Prediction", "Green_Probability", "lead_label"]]
                               ].head(50), use_container_width=True)

        # Download button
        csv_out = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Scored Leads CSV",
            data=csv_out,
            file_name="scored_leads.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Make sure your CSV has the same column names as the training data.")


# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#95a5a6; font-size:0.85rem;'>"
    "🏠 Property Lead Scoring AI &nbsp;|&nbsp; Built with Random Forest + Streamlit"
    "</div>",
    unsafe_allow_html=True
)