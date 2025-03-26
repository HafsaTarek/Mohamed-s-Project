import streamlit as st
import joblib
import pandas as pd
import tempfile
import numpy as np

# Load model and other necessary objects
model = joblib.load('xgb_model.pkl')
selector = joblib.load('selector.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Features list
selected_features = [
    'العام المالي', 'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 
    'معدل دوران الاصول', 'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 
    'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
]

# Customize UI and Colors
st.set_page_config(page_title="Credit Scoring Prediction", page_icon="📊", layout="wide")

# Page Title with custom style
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #0B3954;
            font-size: 36px;
            font-weight: bold;
        }
    </style>
    <div class="title">🏦 Credit Category Prediction 🏦</div>
""", unsafe_allow_html=True)

# File uploader for Excel file
uploaded_file = st.file_uploader("📂 **Upload the Excel file containing financial data**", type=['xlsx'])

if uploaded_file is not None:
    with st.spinner("Processing the data..."):
        try:
            # Read the uploaded Excel file
            df = pd.read_excel(uploaded_file)

            # Clean column names
            df.columns = df.columns.str.strip()

            # Check if the required columns are present in the file
            if not all(col in df.columns for col in selected_features):
                st.error("❌ The file does not contain all the required columns!")
            else:
                # Convert percentage columns to decimal format
                percent_cols = ['صافي الربح', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI']
                for col in percent_cols:
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100

                # Ensure the columns have numeric data types
                for col in selected_features:
                    if df[col].dtype not in ['float64', 'int64']:
                        st.error(f"❌ The column {col} should be numeric!")

                # Fill missing values with median
                df.fillna(df.median(numeric_only=True), inplace=True)

                # Prepare the data for prediction
                input_data_scaled = scaler.transform(df[selected_features])
                input_data_selected = selector.transform(input_data_scaled)

                # Perform predictions
                predictions = model.predict(input_data_selected)
                df["Predicted Class"] = label_encoder.inverse_transform(predictions)

                # Display the results in the same order as the input
                st.write("### 🔍 **Prediction Results**")
                st.dataframe(df)  # Show data in the same order as the file

                # Save the results to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                    df.to_excel(tmp_file.name, index=False)
                    with open(tmp_file.name, "rb") as file:
                        st.download_button("📥 Download Results", file, file_name="predicted_results.xlsx")

        except Exception as e:
            st.error(f"🚨 Error: {e}")

# Inputs for new prediction
st.write("### 🔍 **Predict using New Data**")
input_values = {}

# Create input fields for each feature
# Create input fields for each feature
for feature in selected_features:
    # For percentage features
    if 'نسبة' in feature or 'العائد' in feature:
        input_values[feature] = st.number_input(feature, format="%.5f")  # Accepts any value, with 5 decimal precision
    else:
        input_values[feature] = st.number_input(feature, format="%.5f")  # Accepts any value, with 5 decimal precision


# Prediction for new inputs
if st.button("Predict"):
    try:
        # Convert inputs into DataFrame
        input_df = pd.DataFrame([input_values])

        # Prepare the data for prediction
        input_data_scaled = scaler.transform(input_df)
        input_data_selected = selector.transform(input_data_scaled)

        # Perform prediction
        prediction = model.predict(input_data_selected)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        st.success(f"📊 **Predicted Class:** {predicted_class}")
    except Exception as e:
        st.error(f"🚨 Error: {e}")

