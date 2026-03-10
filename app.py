import streamlit as st
import pandas as pd
import joblib
import random

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="EquiCorp Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

#st.session_state["_sidebar_open"] = True

# ----------------------------
# Load model artifacts
# ----------------------------
model = joblib.load("uci_rf_model.pkl")
columns = joblib.load("uci_columns.pkl")

THRESHOLD = 0.4

# ----------------------------
# Session state
# ----------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "probability" not in st.session_state:
    st.session_state.probability = None

if "decision" not in st.session_state:
    st.session_state.decision = None

if "application_status" not in st.session_state:
    st.session_state.application_status = "Pending Review"

if "current_page" not in st.session_state:
    st.session_state.current_page = "Advisor Dashboard"

if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "applicant_name": "Amina Okafor",
        "application_id": "EQ-2038",

        # UCI core fields
        "limit_bal": 12000,
        "sex": "Female",
        "education": "University",
        "marriage": "Single",
        "age": 29,

        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,

        "bill_amt1": 2500,
        "bill_amt2": 2200,
        "bill_amt3": 2100,
        "bill_amt4": 2000,
        "bill_amt5": 1800,
        "bill_amt6": 1600,

        "pay_amt1": 500,
        "pay_amt2": 500,
        "pay_amt3": 450,
        "pay_amt4": 400,
        "pay_amt5": 350,
        "pay_amt6": 300,

        # UI-only / advisor notes
        "employment_status": "Full-Time",
        "monthly_income": 4200,
        "review_note": ""
    }

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
    /* Remove Streamlit's default top padding so content isn't hidden */
    .appview-container .main .block-container {
        padding-top: 0rem;
        max-width: 1200px;
        padding-bottom: 2rem;
    }
    .reportview-container .main .block-container {
        padding-top: 0rem;
        max-width: 1200px;
        padding-bottom: 2rem;
    }

    /* Top header */
    .brand {
        font-size: 1.5rem;
        font-weight: 800;
        color: #2563eb;
        letter-spacing: -0.03em;
    }

    .brand-sub {
        color: rgba(128,128,128,0.9);
        font-size: 0.95rem;
    }

    .form-section-card {
        background: var(--secondary-background-color);
        border-radius: 18px;
        padding: 1.2rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 1rem;
    }

    .form-section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #2563eb;
    }

    .form-section-sub {
        font-size: 0.88rem;
        color: rgba(148,163,184,0.9);
        margin-bottom: 0.9rem;
        line-height: 1.5;
    }

    .helper-note {
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        font-size: 0.87rem;
        color: inherit;
        margin-top: 0.75rem;
    }

    .submit-wrap {
        text-align: center;
        margin-top: 1.2rem;
    }
            
    .top-cta-btn {
        background: #1d4ed8;
        color: white;
        padding: 0.6rem 1.4rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Sidebar system status card */
    .sidebar-status {
        background: #ecfdf3;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        border: 1px solid #bbf7d0;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }

    .sidebar-status-title {
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #14532d;
    }

    .sidebar-status-sub {
        color: #166534;
        font-size: 0.85rem;
    }

    .page-card {
        background: var(--secondary-background-color);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: inherit;
        margin-bottom: 0.1rem;
    }

    .subtle {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }

    .metric-box {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(15, 23, 42, 0.05);
        min-height: 110px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: inherit;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .status-pill-good {
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }

    .status-pill-bad {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }

    .mini-card {
        background: var(--secondary-background-color);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 0.8rem;
    }

    .mini-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }

    .mini-meta {
        color: rgba(148,163,184,0.9);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .blue-note {
        background: #eff6ff;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border: 1px solid #bfdbfe;
        font-size: 0.88rem;
        color: #1e3a8a;
        line-height: 1.6;
    }

    .result-card {
        background: var(--secondary-background-color);
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(15, 23, 42, 0.06);
        min-height: 100%;
    }

    .summary-row {
        color: inherit;
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
    }

    .summary-row strong {
        color: rgba(128,128,128,0.9);
        margin-right: 0.25rem;
    }

    .table-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
    }

    .risk-pill-low {
        background: #ecfdf3;
        color: #166534;
        border-radius: 999px;
        padding: 0.1rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .risk-pill-medium {
        background: #fffbeb;
        color: #92400e;
        border-radius: 999px;
        padding: 0.1rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
            
    /* Top header action buttons only */
    div[data-testid="stButton"] button[kind="secondary"] {
        border-radius: 999px;
        font-weight: 600;
    }

/* Start New Application button */
    div[data-testid="stButton"] button:has(div[data-testid="stMarkdownContainer"] p:contains("Start New Application")) {
        background-color: #1d4ed8;
        color: white;
        border: none;
    }

/* Dashboard button */
    div[data-testid="stButton"] button:has(div[data-testid="stMarkdownContainer"] p:contains("Dashboard")) {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    } 
    
    .stButton > button[kind="secondary"] {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #1d4ed8;
    }
            
    .risk-pill-high {
        background: #fef2f2;
        color: #b91c1c;
        border-radius: 999px;
        padding: 0.1rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def run_prediction(form_data):
    # map UI labels to dataset codes
    sex_map = {
        "Male": 1,
        "Female": 2
    }

    education_map = {
        "Graduate School": 1,
        "University": 2,
        "High School": 3,
        "Other": 4
    }

    marriage_map = {
        "Married": 1,
        "Single": 2,
        "Other": 3
    }

    # build one row exactly like the UCI training data
    input_dict = {
        "LIMIT_BAL": form_data["limit_bal"],
        "SEX": sex_map[form_data["sex"]],
        "EDUCATION": education_map[form_data["education"]],
        "MARRIAGE": marriage_map[form_data["marriage"]],
        "AGE": form_data["age"],

        "PAY_0": form_data["pay_0"],
        "PAY_2": form_data["pay_2"],
        "PAY_3": form_data["pay_3"],
        "PAY_4": form_data["pay_4"],
        "PAY_5": form_data["pay_5"],
        "PAY_6": form_data["pay_6"],

        "BILL_AMT1": form_data["bill_amt1"],
        "BILL_AMT2": form_data["bill_amt2"],
        "BILL_AMT3": form_data["bill_amt3"],
        "BILL_AMT4": form_data["bill_amt4"],
        "BILL_AMT5": form_data["bill_amt5"],
        "BILL_AMT6": form_data["bill_amt6"],

        "PAY_AMT1": form_data["pay_amt1"],
        "PAY_AMT2": form_data["pay_amt2"],
        "PAY_AMT3": form_data["pay_amt3"],
        "PAY_AMT4": form_data["pay_amt4"],
        "PAY_AMT5": form_data["pay_amt5"],
        "PAY_AMT6": form_data["pay_amt6"],
    }

    input_df = pd.DataFrame([input_dict])

    # align with training columns exactly
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]
    decision = "High Risk Applicant" if prob >= THRESHOLD else "Low Risk Applicant"

    st.session_state.probability = prob
    st.session_state.decision = decision
    st.session_state.prediction_done = True
    st.session_state.application_status = "Completed"

# ----------------------------
# Sidebar navigation (left nav like design)
# ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-status">
            <div class="sidebar-status-title">System Status</div>
            <div class="sidebar-status-sub">
                Fairness monitoring enabled. AI‑driven bias detection is active for all evaluations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Advisor Dashboard", "Loan Application Form", "Results Page"],
        label_visibility="collapsed",
        index=["Advisor Dashboard", "Loan Application Form", "Results Page"].index(
            st.session_state.current_page
        ),
    )

st.session_state.current_page = page

# ----------------------------
# Top header bar
# ----------------------------
def top_button_style(label, bg, fg, border):
    st.markdown(
        f"""
        <style>
        div.stButton > button {{
            border-radius: 999px;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">EquiCorp Advisor Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-sub">Manage and evaluate incoming loan applications with built‑in fairness monitoring.</div>',
        unsafe_allow_html=True,
    )

with header_right:
    top1, top2, top3 = st.columns([1.1, 1.2, 1.2])

    with top1:
        st.markdown(
            '<div style="font-size:0.9rem;color:#6b7280;padding-top:0.45rem;">John Doe (Advisor)</div>',
            unsafe_allow_html=True,
        )

    with top2:
        if st.button("Dashboard", key="dashboard_btn"):
            st.session_state.current_page = "Advisor Dashboard"
            st.rerun()

    with top3:
        if st.button("Start New Application", key="new_app_btn"):

            new_id = f"EQ-{random.randint(1000,9999)}"

            st.session_state.form_data = {
                "applicant_name": "",
                "application_id": new_id,
                "limit_bal": 0,
                "sex": "Select",
                "education": "Select",
                "marriage": "Select",
                "age": 18,

                "pay_0": 0,
                "pay_2": 0,
                "pay_3": 0,
                "pay_4": 0,
                "pay_5": 0,
                "pay_6": 0,

                "bill_amt1": 0,
                "bill_amt2": 0,
                "bill_amt3": 0,
                "bill_amt4": 0,
                "bill_amt5": 0,
                "bill_amt6": 0,

                "pay_amt1": 0,
                "pay_amt2": 0,
                "pay_amt3": 0,
                "pay_amt4": 0,
                "pay_amt5": 0,
                "pay_amt6": 0,

                "employment_status": "Select",
                "monthly_income": 0,
                "review_note": ""
            }

            st.session_state.prediction_done = False
            st.session_state.probability = None
            st.session_state.decision = None
            st.session_state.current_page = "Loan Application Form"

            st.rerun()

st.write("")

# ----------------------------
# Advisor Dashboard
# ----------------------------
if page == "Advisor Dashboard":
    stats_left, stats_mid, stats_right = st.columns(3)

    prob_display = "--"
    decision_display = "Pending Review"
    if st.session_state.prediction_done:
        prob_display = f"{st.session_state.probability:.2f}"
        decision_display = st.session_state.decision

    with stats_left:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Default Probability</div>
                <div class="metric-value">{prob_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stats_mid:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Decision Status</div>
                <div class="metric-value" style="font-size:1.2rem;">{decision_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stats_right:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">Applications Completed</div>
                <div class="metric-value">12</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Recent applications table card
    st.markdown('<div class="page-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Recent Applications</div>'
        '<div class="subtle">Manage and evaluate incoming loan applications.</div>',
        unsafe_allow_html=True,
    )

    recent_apps = pd.DataFrame({
        "Application ID": ["EQ-2038", "EQ-2041", "EQ-2046", "EQ-2050"],
        "Client Name": ["Amina Okafor", "David Mensah", "Ngozi Peter", "Samuel Bello"],
        "Risk Level": ["Medium", "Low", "High", "Medium"],
        "Status": ["Pending Review", "Reviewed", "Submitted", "Awaiting Decision"],
        "Date Submitted": ["2026-03-08", "2026-03-07", "2026-03-07", "2026-03-06"],
    })
    st.dataframe(
        recent_apps,
        use_container_width=True,
        hide_index=True,
        height=220
    )

    st.markdown(
        '<div style="text-align:center;font-size:0.85rem;color:#1d4ed8;font-weight:500;margin-top:0.6rem;">Show All Applications</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Bottom help cards
    help_left, help_right = st.columns(2)

    with help_left:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">📘 Need Help?</div>
                <div class="mini-meta">
                    Check our guide on how to interpret high‑risk fairness flags in automated evaluations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with help_right:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-title">📊 Review Logs</div>
                <div class="mini-meta">
                    Audit trails for all advisor decisions are automatically saved for compliance review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# ----------------------------
# Loan Application Form
# ----------------------------
elif page == "Loan Application Form":
    st.markdown('<div class="page-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Loan Application Form</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Complete the applicant profile below and submit the case for evaluation.</div>',
        unsafe_allow_html=True
    )
    st.write("")

    data = st.session_state.form_data

    st.markdown("### Applicant Information")
    a1, a2 = st.columns(2)
    with a1:
        applicant_name = st.text_input("Applicant Name", value=data.get("applicant_name", ""))
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(data.get("age", 18))
        )
        sex_options = ["Select", "Male", "Female"]
        sex = st.selectbox(
            "Sex",
            sex_options,
            index=sex_options.index(data.get("sex", "Select"))
        )
        marriage_options = ["Select", "Married", "Single", "Other"]
        marriage = st.selectbox(
            "Marital Status",
            marriage_options,
            index=marriage_options.index(data.get("marriage", "Select"))
        )

    with a2:
        application_id = st.text_input(
            "Application ID",
            value=data.get("application_id", ""), disabled=True
        )
        education_options = ["Select", "Graduate School", "University", "High School", "Other"]
        education = st.selectbox(
                    "Education Level",
                    education_options,
                    index=education_options.index(data.get("education", "Select"))
        )
        employment_options = ["Select", "Full-Time", "Part-Time", "Self-Employed", "Unemployed"]
        employment_status = st.selectbox(
            "Employment Status",
            employment_options,
            index=employment_options.index(data.get("employment_status", "Select"))
        )
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0,
            value=int(data.get("monthly_income", 0))
        )

    st.write("")
    st.markdown("### Loan Details")
    b1, b2 = st.columns(2)
    with b1:
        limit_bal = st.number_input(
            "Credit Limit (LIMIT_BAL)",
            min_value=0,
            value=int(data.get("limit_bal", 0))
        )
    with b2:
        review_note = st.text_area(
            "Advisor Notes",
            value=data.get("review_note", ""),
            placeholder="Add any additional notes relevant to the applicant..."
        )

    st.write("")

    with st.expander("Advanced Financial History (Optional)"):

        st.markdown("#### Repayment Status History")
        st.markdown(
            '<div class="subtle">Repayment status from the past 6 months. Typical values: -1 = paid on time, 0 = use of revolving credit, 1+ = payment delay.</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            pay_0 = st.number_input("PAY_0 (Last Month)")
            pay_2 = st.number_input("PAY_2 (2 Months Ago)")

        with c2:
            pay_3 = st.number_input("PAY_3 (3 Months Ago)")
            pay_4 = st.number_input("PAY_4 (4 Months Ago)")

        with c3:
            pay_5 = st.number_input("PAY_5 (5 Months Ago)")
            pay_6 = st.number_input("PAY_6 (6 Months Ago)")

        st.write("")

        st.markdown("#### Bill Statement History")

        d1, d2, d3 = st.columns(3)

        with d1:
            bill_amt1 = st.number_input("BILL_AMT1 (Last Month Bill)")
            bill_amt2 = st.number_input("BILL_AMT2")

        with d2:
            bill_amt3 = st.number_input("BILL_AMT3")
            bill_amt4 = st.number_input("BILL_AMT4")

        with d3:
            bill_amt5 = st.number_input("BILL_AMT5")
            bill_amt6 = st.number_input("BILL_AMT6")

        st.write("")

        st.markdown("#### Payment Amount History")

        e1, e2, e3 = st.columns(3)

        with e1:
            pay_amt1 = st.number_input("PAY_AMT1 (Last Payment)")
            pay_amt2 = st.number_input("PAY_AMT2")

        with e2:
            pay_amt3 = st.number_input("PAY_AMT3")
            pay_amt4 = st.number_input("PAY_AMT4")

        with e3:
            pay_amt5 = st.number_input("PAY_AMT5")
            pay_amt6 = st.number_input("PAY_AMT6")

    submit = st.button("Analyze Application", use_container_width=True)

    if submit:
        st.session_state.form_data = {
            "applicant_name": applicant_name,
            "application_id": application_id,
            "age": age,
            "sex": sex,
            "education": education,
            "marriage": marriage,
            "limit_bal": limit_bal,
            "employment_status": employment_status,
            "monthly_income": monthly_income,
            "review_note": review_note,

            "pay_0": pay_0,
            "pay_2": pay_2,
            "pay_3": pay_3,
            "pay_4": pay_4,
            "pay_5": pay_5,
            "pay_6": pay_6,

            "bill_amt1": bill_amt1,
            "bill_amt2": bill_amt2,
            "bill_amt3": bill_amt3,
            "bill_amt4": bill_amt4,
            "bill_amt5": bill_amt5,
            "bill_amt6": bill_amt6,

            "pay_amt1": pay_amt1,
            "pay_amt2": pay_amt2,
            "pay_amt3": pay_amt3,
            "pay_amt4": pay_amt4,
            "pay_amt5": pay_amt5,
            "pay_amt6": pay_amt6
        }

        run_prediction(st.session_state.form_data)
        st.session_state.current_page = "Results Page"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Results Page
# ----------------------------
else:
    st.markdown('<div class="page-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results Page</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Final review output for the submitted application.</div>', unsafe_allow_html=True)
    st.write("")

    if not st.session_state.prediction_done:
        st.info("No evaluation available yet. Please complete the Loan Application Form first.")
    else:
        data = st.session_state.form_data
        left, right = st.columns([1.1, 1], gap="large")

        with left:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### Applicant Summary")
            st.markdown(f'<div class="summary-row"><strong>Application ID:</strong> {data["application_id"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Name:</strong> {data["applicant_name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Age:</strong> {data["age"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Sex:</strong> {data["sex"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Education:</strong> {data["education"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Marital Status:</strong> {data["marriage"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Credit Limit:</strong> {data["limit_bal"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Employment Status:</strong> {data["employment_status"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Monthly Income:</strong> {data["monthly_income"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Most Recent Bill Amount:</strong> {data["bill_amt1"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-row"><strong>Most Recent Payment Amount:</strong> {data["pay_amt1"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")

        st.markdown("#### Credit Behavior Indicators")

        c1, c2 = st.columns(2)

        with c1:
            st.write("Repayment Status (PAY_0):", data["pay_0"])
            st.write("Repayment Status (PAY_2):", data["pay_2"])
            st.write("Repayment Status (PAY_3):", data["pay_3"])

        with c2:
            st.write("Recent Bill Amount:", data["bill_amt1"])
            st.write("Recent Payment Amount:", data["pay_amt1"])

        with right:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### Risk Outcome")
            st.markdown(f"""
                <div class="metric-box" style="margin-bottom:1rem;">
                    <div class="metric-label">Default Probability</div>
                    <div class="metric-value">{st.session_state.probability:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

            if st.session_state.decision == "Low Risk Applicant":
                st.success("Approved as Low Risk Applicant")
            else:
                st.error("Flagged as High Risk Applicant")

            st.write("")
            st.markdown("### Fairness Review")
            st.markdown("""
                <div class="blue-note">
                    EquiCorp’s evaluation workflow includes fairness monitoring to reduce the risk of biased lending outcomes.
                    The model was assessed using demographic fairness metrics across sex and age-group comparisons,
                    allowing the team to review not only predictive accuracy, but also whether decision patterns remain
                    balanced across sensitive applicant groups.
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("### Advisor Summary")
            if st.session_state.decision == "Low Risk Applicant":
                st.write("This applicant falls below the current review threshold and may proceed to the next approval stage.")
            else:
                st.write("This applicant exceeds the current review threshold and should be flagged for additional human review.")

            st.markdown("### Model Confidence")
            st.progress(float(st.session_state.probability))
            st.write(f"Default Probability: {st.session_state.probability:.2f}")

            if st.session_state.probability < 0.3:
                st.write("Low probability of default. Applicant profile aligns with historically approved borrowers.")
            elif st.session_state.probability < 0.5:
                st.write("Moderate probability. Additional manual review recommended.")
            else:
                st.write("Elevated probability of default. Applicant flagged for risk review.")

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    if st.button("Evaluate Another Application"):
        st.session_state.current_page = "Loan Application Form"
        st.session_state.prediction_done = False
        st.rerun()