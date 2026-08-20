
import streamlit as st
import pandas as pd
import joblib

# ---------- Load models ----------
# Extra Trees + its Scaler (these should always load successfully)
extra_trees = joblib.load("extra_trees.pkl")
scaler = joblib.load("scaler.pkl")

# Logistic Regression Pipeline (self-contained: includes its own scaling/encoding)
# Loading is wrapped in try/except so the app still works even if this file
# hasn't been re-saved with a matching scikit-learn version yet.
try:
    logistic_model = joblib.load("logistic_regression.pkl")
    logistic_available = True
except Exception as e:
    logistic_model = None
    logistic_available = False
    logistic_error = str(e)

# Random Forest (assumed trained the same way as Extra Trees: same
# one-hot encoded columns + same scaler). Wrapped in try/except so the
# app still works if random_forest.pkl isn't in the folder yet.
try:
    random_forest = joblib.load("random_forest.pkl")
    rf_available = True
except Exception as e:
    random_forest = None
    rf_available = False
    rf_error = str(e)

# Exact column order/names the Extra Trees model + Scaler were trained on
EXTRA_TREES_COLUMNS = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
    'job_retired', 'job_self-employed', 'job_services', 'job_student',
    'job_technician', 'job_unemployed', 'job_unknown',
    'marital_married', 'marital_single',
    'education_secondary', 'education_tertiary', 'education_unknown',
    'default_yes', 'housing_yes', 'loan_yes',
    'contact_telephone', 'contact_unknown',
    'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',
    'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
    'poutcome_other', 'poutcome_success', 'poutcome_unknown'
]


def encode_for_extra_trees(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode raw input the same way it was done during training
    (pd.get_dummies with drop_first=True), then align columns exactly
    to what the scaler/model expect, filling any missing dummy columns with 0."""
    encoded = pd.get_dummies(raw_df, drop_first=True)
    encoded = encoded.reindex(columns=EXTRA_TREES_COLUMNS, fill_value=0)
    return encoded


st.set_page_config(page_title="Bank Deposit Prediction", page_icon="🏦")
st.title("🏦 Bank Term Deposit Prediction")
st.write("Fill in the customer details below, then click a Predict button.")

st.divider()

# ---------- Inputs (shared by both models) ----------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    balance = st.number_input("Account Balance", value=1000)
    housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

with col2:
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
    day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=180)
    campaign = st.number_input("Number of Contacts in This Campaign", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact (previous campaign, -1 = never contacted)", value=-1)
    previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, value=0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "failure", "other", "success"])

st.divider()

raw_input = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}])

# ---------- Model selection ----------
model_choice = st.radio(
    "Choose a model",
    ["Logistic Regression", "Extra Trees", "Random Forest"],
    horizontal=True,
)

if st.button("Predict 🔍", use_container_width=True):
    if model_choice == "Logistic Regression":
        if not logistic_available:
            st.warning(
                "⚠️ The Logistic Regression model file couldn't be loaded "
                "(version mismatch). Please re-save it from Colab with the "
                "matching scikit-learn version, then replace "
                "logistic_regression.pkl and rerun."
            )
            st.caption(f"Technical detail: {logistic_error}")
        else:
            prediction = logistic_model.predict(raw_input)[0]
            probability = logistic_model.predict_proba(raw_input)[0][1]
            if prediction == 1:
                st.success(f"✅ The customer is likely to subscribe to a term deposit (confidence: {probability:.1%})")
            else:
                st.error(f"❌ The customer is not likely to subscribe to a term deposit (subscription probability: {probability:.1%})")

    elif model_choice == "Extra Trees":
        encoded_input = encode_for_extra_trees(raw_input)
        scaled_input = scaler.transform(encoded_input)
        prediction = extra_trees.predict(scaled_input)[0]
        probability = extra_trees.predict_proba(scaled_input)[0][1]
        if prediction == 1:
            st.success(f"✅ The customer is likely to subscribe to a term deposit (confidence: {probability:.1%})")
        else:
            st.error(f"❌ The customer is not likely to subscribe to a term deposit (subscription probability: {probability:.1%})")

    else:  # Random Forest
        if not rf_available:
            st.warning(
                "⚠️ The Random Forest model file couldn't be loaded. "
                "Make sure random_forest.pkl is in the same folder as this app."
            )
            st.caption(f"Technical detail: {rf_error}")
        else:
            encoded_input = encode_for_extra_trees(raw_input)
            scaled_input = scaler.transform(encoded_input)
            prediction = random_forest.predict(scaled_input)[0]
            probability = random_forest.predict_proba(scaled_input)[0][1]
            if prediction == 1:
                st.success(f"✅ The customer is likely to subscribe to a term deposit (confidence: {probability:.1%})")
            else:
                st.error(f"❌ The customer is not likely to subscribe to a term deposit (subscription probability: {probability:.1%})")