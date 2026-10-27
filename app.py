from pathlib import Path

import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# --------------------------------------------------
# ChurnIQ - Customer Retention Intelligence
# Uses the trained model and preprocessing from the
# original Customer Churn Prediction Jupyter notebook.
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "best_model.pkl"
ENCODER_PATH = APP_DIR / "encoder.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"
DATA_PATH = APP_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1020, #10182b 60%, #111326);
    color: #f2f4ff;
}
[data-testid="stSidebar"] {
    background: #101629;
    border-right: 1px solid #29334d;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #35365f;
    background: linear-gradient(115deg, #1c2441, #27204b, #152a43);
    margin-bottom: 22px;
}
.eyebrow {
    color: #82e7e2;
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 700;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin: 8px 0;
}
.hero p {
    color: #c4cce4;
    margin-bottom: 0;
}
.metric-card {
    background: #171f34;
    border: 1px solid #303b58;
    padding: 20px;
    border-radius: 16px;
    min-height: 112px;
}
.metric-label {
    color: #aebbd8;
    font-size: 13px;
}
.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 700;
    margin-top: 9px;
}
.section-card {
    background: #141d31;
    border: 1px solid #303b58;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
h2, h3 {
    color: #f4f5ff;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(100deg, #8873ff, #5688ff);
    color: white;
    border: 0;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border: 1px solid #a99cff;
    color: white;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #161f34;
}
div[data-testid="stForm"] {
    border: 1px solid #303b58;
    border-radius: 16px;
    padding: 20px;
    background: #111a2d;
}
.small-muted {
    color: #aebbd8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, ENCODER_PATH, SCALER_PATH]
    missing = [p.name for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required model file(s): " + ", ".join(missing)
        )

    # Only load pickle files created by your own trusted notebook.
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, encoders, scaler


try:
    model, encoders, scaler = load_artifacts()
    artifacts_ready = True
    load_error = ""
except Exception as exc:
    artifacts_ready = False
    load_error = str(exc)
    model = encoders = scaler = None


with st.sidebar:
    st.markdown("## ✦ ChurnIQ")
    st.caption("CUSTOMER RETENTION INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["New Prediction", "Portfolio Insights", "About"],
        label_visibility="visible"
    )

    st.divider()

    if artifacts_ready:
        st.success("Model files loaded")
    else:
        st.error("Model files need attention")
        st.caption(load_error)

    st.markdown(
        "<div class='small-muted'>Machine learning · Customer analytics</div>",
        unsafe_allow_html=True
    )


st.markdown("""
<div class="hero">
    <div class="eyebrow">CUSTOMER RETENTION INTELLIGENCE</div>
    <h1>ChurnIQ <span style="color:#9d8bff;">✦</span></h1>
    <p>Understand customer risk. Make smarter retention decisions.</p>
</div>
""", unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        </div>""",
        unsafe_allow_html=True
    )


def predict_customer(customer):
    """Apply the same encoders and scaler used in the notebook."""
    X = pd.DataFrame([customer], columns=FEATURE_ORDER)

    # The notebook trained a separate LabelEncoder for each
    # categorical column. Preserve the exact saved category mapping.
    for col, encoder in encoders.items():
        if col not in X.columns:
            continue

        values = X[col].astype(str)
        known = set(map(str, encoder.classes_))
        unknown = [v for v in values if v not in known]

        if unknown:
            raise ValueError(
                f"Unknown category '{unknown[0]}' for {col}. "
                "Choose a category present in the original dataset."
            )

        X[col] = encoder.transform(values)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    # Preserve the original feature order from the notebook.
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        if set(expected) != set(X.columns):
            raise ValueError(
                "The model's expected input columns differ from the app's "
                "feature columns. Check the training features."
            )
        X = X[expected]

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)
    churn_positions = [
        i for i, cls in enumerate(classes)
        if str(cls) == "1"
    ]

    if not churn_positions:
        raise ValueError(
            f"Could not identify the churn class in model classes: {classes}"
        )

    churn_probability = float(probabilities[churn_positions[0]])
    return int(prediction), churn_probability


if page == "New Prediction":
    st.subheader("New customer assessment")
    st.markdown(
        "<p class='small-muted'>Enter a customer's profile and billing "
        "information to estimate churn risk.</p>",
        unsafe_allow_html=True
    )

    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("### Customer profile")
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
            senior = st.selectbox(
                "Senior citizen", ["No", "Yes"]
            )
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox(
                "Dependents", CATEGORICAL_OPTIONS["Dependents"]
            )
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100, value=12
            )
            phone = st.selectbox(
                "Phone service", CATEGORICAL_OPTIONS["PhoneService"]
            )
            multiple = st.selectbox(
                "Multiple lines", CATEGORICAL_OPTIONS["MultipleLines"]
            )

        with col2:
            st.markdown("### Internet & services")
            internet = st.selectbox(
                "Internet service", CATEGORICAL_OPTIONS["InternetService"]
            )
            security = st.selectbox(
                "Online security", CATEGORICAL_OPTIONS["OnlineSecurity"]
            )
            backup = st.selectbox(
                "Online backup", CATEGORICAL_OPTIONS["OnlineBackup"]
            )
            protection = st.selectbox(
                "Device protection", CATEGORICAL_OPTIONS["DeviceProtection"]
            )
            support = st.selectbox(
                "Tech support", CATEGORICAL_OPTIONS["TechSupport"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", CATEGORICAL_OPTIONS["StreamingMovies"]
            )

        with col3:
            st.markdown("### Billing details")
            contract = st.selectbox(
                "Contract", CATEGORICAL_OPTIONS["Contract"]
            )
            paperless = st.selectbox(
                "Paperless billing", CATEGORICAL_OPTIONS["PaperlessBilling"]
            )
            payment = st.selectbox(
                "Payment method", CATEGORICAL_OPTIONS["PaymentMethod"]
            )
            monthly = st.number_input(
                "Monthly charges ($)", min_value=0.0,
                max_value=1000.0, value=70.0, step=1.0
            )
            total = st.number_input(
                "Total charges ($)", min_value=0.0,
                max_value=100000.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button(
            "✦  Predict churn risk",
            use_container_width=True
        )

    if submitted:
        if not artifacts_ready:
            st.error(
                "The model files could not be loaded. Check the message "
                "in the sidebar and confirm the three .pkl files are beside app.py."
            )
        elif not (tenure * monthly >= 0 and total >= 0):
            st.error("Please check the billing values.")
        else:
            customer = {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total
            }

            try:
                prediction, probability = predict_customer(customer)
                is_churn = prediction == 1
                risk = (
                    "HIGH" if probability >= 0.70
                    else "MEDIUM" if probability >= 0.35
                    else "LOW"
                )
                risk_colors = {
                    "HIGH": "#ff718b",
                    "MEDIUM": "#ffc46b",
                    "LOW": "#63ddb0"
                }

                st.divider()
                st.subheader("Prediction report")

                r1, r2, r3 = st.columns(3)
                with r1:
                    metric_card(
                        "MODEL PREDICTION",
                        "Likely to churn" if is_churn else "Likely to stay"
                    )
                with r2:
                    metric_card("CHURN PROBABILITY", f"{probability:.1%}")
                with r3:
                    metric_card("RISK CATEGORY", risk)

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%",
                        "font": {"size": 38, "color": "#ffffff"}
                    },
                    title={
                        "text": "ESTIMATED CHURN PROBABILITY",
                        "font": {"size": 15, "color": "#c3cce4"}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "ticksuffix": "%",
                            "tickcolor": "#aebbd8"
                        },
                        "bar": {
                            "color": "#9381ff",
                            "thickness": 0.28
                        },
                        "bgcolor": "#171f34",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 35], "color": "#173f3c"},
                            {"range": [35, 70], "color": "#554326"},
                            {"range": [70, 100], "color": "#542b3b"}
                        ]
                    }
                ))
                gauge.update_layout(
                    height=340,
                    paper_bgcolor="#0f1729",
                    font_color="white",
                    margin=dict(l=25, r=25, t=65, b=20)
                )

                left, right = st.columns([1.15, 1])
                with left:
                    st.plotly_chart(gauge, use_container_width=True)

                with right:
                    st.markdown("### Recommended next step")
                    if risk == "HIGH":
                        recommendation = (
                            "Prioritise a retention review. Consider contacting "
                            "the customer to understand concerns and discuss "
                            "suitable contract or service options."
                        )
                    elif risk == "MEDIUM":
                        recommendation = (
                            "Monitor engagement and consider a relevant "
                            "service check-in or renewal offer."
                        )
                    else:
                        recommendation = (
                            "Continue good service and regular engagement. "
                            "A low predicted risk does not guarantee retention."
                        )

                    st.markdown(
                        f"""<div class="section-card">
                        <div style="font-size:12px;letter-spacing:2px;
                        color:{risk_colors[risk]};font-weight:700;">
                        {risk} RISK
                        </div>
                        <h3>{'Retention attention suggested' if is_churn else 'Retention maintenance'}</h3>
                        <p style="color:#c3cce4;">{recommendation}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )

                st.caption(
                    "This is a machine-learning estimate based on the "
                    "provided customer profile, not a guarantee of future behaviour."
                )

            except Exception as exc:
                st.error("Prediction could not be completed.")
                st.exception(exc)


elif page == "Portfolio Insights":
    st.subheader("Portfolio insights")

    if not DATA_PATH.exists():
        st.warning(
            "Dataset CSV not found beside app.py. Add the original "
            "WA_Fn-UseC_-Telco-Customer-Churn.csv file to enable these charts."
        )
    else:
        try:
            df = pd.read_csv(DATA_PATH)
            total_customers = len(df)
            churn_count = (df["Churn"].astype(str).str.lower() == "yes").sum()
            churn_rate = churn_count / total_customers if total_customers else 0
            avg_monthly = pd.to_numeric(
                df["MonthlyCharges"], errors="coerce"
            ).mean()

            a, b, c = st.columns(3)
            with a:
                metric_card("CUSTOMERS", f"{total_customers:,}")
            with b:
                metric_card("HISTORICAL CHURN RATE", f"{churn_rate:.1%}")
            with c:
                metric_card("AVG. MONTHLY CHARGES", f"${avg_monthly:.2f}")

            st.markdown("### Churn by contract type")
            chart_data = (
                df.groupby(["Contract", "Churn"])
                .size().reset_index(name="Customers")
            )
            fig = go.Figure()
            for status in ["Yes", "No"]:
                subset = chart_data[chart_data["Churn"] == status]
                fig.add_bar(
                    x=subset["Contract"],
                    y=subset["Customers"],
                    name="Churned" if status == "Yes" else "Stayed"
                )
            fig.update_layout(
                barmode="group",
                paper_bgcolor="#0f1729",
                plot_bgcolor="#0f1729",
                font_color="#e6eaff",
                legend_title="Customer status",
                xaxis_title="Contract",
                yaxis_title="Number of customers"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error("Could not read the portfolio dataset.")
            st.exception(exc)


else:
    st.subheader("About ChurnIQ")
    st.markdown("""
    <div class="section-card">
        <h3>Customer Churn Prediction System</h3>
        <p style="color:#c3cce4;">
        ChurnIQ uses a previously trained machine-learning model to estimate
        whether a telecom customer is likely to leave the service.
        </p>
        <h3>Technology stack</h3>
        <p style="color:#c3cce4;">
        Python · Pandas · Scikit-learn · Streamlit · Plotly
        </p>
        <h3>Prediction workflow</h3>
        <p style="color:#c3cce4;">
        Customer inputs → saved label encoders → saved numerical scaler →
        trained classifier → churn prediction and probability.
        </p>
    </div>
    """, unsafe_allow_html=True)
