import streamlit as st
st.set_page_config(page_title="Vehicle Transmission Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ==========================
# 📦 Load Data and Model
# ==========================
@st.cache_data
def load_readable_data():
    return pd.read_csv("df_copy_unique.csv")

@st.cache_resource
def load_balanced_data():
    X = pd.read_csv("X_train_balanced.csv")
    y = pd.read_csv("y_train_balanced.csv").squeeze()
    return X, y

@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("logistic_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

df = load_readable_data()
X_train_bal, y_train_bal = load_balanced_data()
model, label_encoders = load_model_and_encoders()
feature_columns = X_train_bal.columns.tolist()

# ==========================
# 🗂️ Tabs Layout (Home + Predict)
# ==========================
tab1, tab2 = st.tabs(["🏠 Home", "🔮 Predict"])

# ==========================
# 🏠 HOME TAB (with background image)
# ==========================
with tab1:
    st.markdown(
        """
        <style>
        .custom-home {
            background-image: url("https://goauto-assets.imgix.net/RV/Go-RV-Edmonton.jpg?auto=format&ixlib=react-9.7.0&w=1075");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 100px 0;
            border-radius: 10px;
        }

        .overlay-text {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            max-width: 900px;
            margin: auto;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        }

        .overlay-text h1 {
            color: #FF5733;
            font-size: 36px;
            margin-bottom: 20px;
        }

        .overlay-text p {
            font-size: 18px;
            font-weight: 500;
            color: #333;
        }
        </style>

        <div class="custom-home">
            <div class="overlay-text">
                <h1>🚗 Vehicle Transmission Prediction App</h1>
                <p>
                    Welcome to the <b style="color:#2E86C1;">Go Auto Machine Learning App</b>!<br><br>
                    This app predicts whether a vehicle has an 
                    <b style="color:#117A65;">Automatic</b> or 
                    <b style="color:#AF601A;">Manual</b> transmission based on key vehicle attributes 
                    like mileage, model, drivetrain, and more.<br><br>
                    <b>Built for:</b> <span style="color: #884EA0;">Go Auto, Edmonton</span><br>
                    <b>Powered by:</b> <span style="color: #CB4335;">Logistic Regression + SMOTE</span>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================
# 🔮 PREDICTION TAB
# ==========================
with tab2:
    st.markdown(
        """
        <style>
        label, .css-1p05t8e, .css-1cpxqw2, .css-1jy4z1n {
            color: #E74C3C !important;
            font-weight: bold !important;
            font-size: 18px !important;
            text-transform: uppercase;
        }

        .css-1d391kg {
            color: #E74C3C !important;
            font-weight: bold !important;
            font-size: 16px !important;
        }

        span[data-testid="stTickBar"] > div {
            color: #E74C3C !important;
            font-weight: bold !important;
            font-size: 16px !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #E74C3C !important;
            font-weight: 700;
        }

        .stMarkdown p {
            color: #000 !important;
            font-weight: 600 !important;
        }

        .stAlert-success {
            background-color: #fdecea;
            color: #E74C3C !important;
            font-weight: bold;
            border-left: 6px solid #E74C3C;
        }

        .stDataFrame div {
            color: #E74C3C !important;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("🔧 Vehicle Input Features")

    def get_user_input():
        user_data = {}
        make = st.selectbox("Make", df["make"].unique())
        models = df[df["make"] == make]["model"].unique()
        model_name = st.selectbox("Model", models)
        series = df[(df["make"] == make) & (df["model"] == model_name)]["series"].unique()
        series_val = st.selectbox("Series", series)

        user_data["make"] = make
        user_data["model"] = model_name
        user_data["series"] = series_val

        for col in feature_columns:
            if col in ["make", "model", "series"]:
                continue
            if df[col].dtype == 'object':
                user_data[col] = st.selectbox(col, df[col].unique())
            else:
                min_val, max_val = int(df[col].min()), int(df[col].max())
                mean_val = int(df[col].mean())
                user_data[col] = st.slider(col, min_val, max_val, mean_val)

        return pd.DataFrame([user_data])

    user_input_df = get_user_input()

    # Encode input
    encoded_input = user_input_df.copy()
    for col in label_encoders:
        if col in encoded_input.columns:
            try:
                encoded_input[col] = label_encoders[col].transform(encoded_input[col])
            except Exception as e:
                st.error(f"Encoding failed for column '{col}': {e}")

    # Fill missing columns with 0
    for col in feature_columns:
        if col not in encoded_input.columns:
            encoded_input[col] = 0

    X_user = encoded_input[feature_columns]

    # Predict
    pred = model.predict(X_user)[0]
    pred_proba = model.predict_proba(X_user)[0]
    label = "Automatic" if pred == 1 else "Manual"

    # Output
    st.subheader("🔍 Summary of Your Input")
    st.write(user_input_df)

    st.subheader("📌 Prediction")
    st.success(f"**Predicted Transmission Type:** {label}")

    st.subheader("📈 Prediction Probability")
    st.dataframe(pd.DataFrame([pred_proba], columns=["Manual", "Automatic"]))





