# # import streamlit as st
# # import joblib
# # import numpy as np
# # import pandas as pd

# # # Load the saved model and preprocessing objects
# # model = joblib.load('xgb_model.pkl')
# # selector = joblib.load('selector.pkl')
# # scaler = joblib.load('scaler.pkl')
# # label_encoder = joblib.load('label_encoder.pkl')

# # selected_features = [
# #     'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 'معدل دوران الاصول',
# #     'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
# # ]


# # # selected_features = [
# # #     'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 'معدل دوران الاصول',
# # #     'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
# # # ]

# # # UI to input new data for prediction
# # st.title("AI Prediction System")
# # st.markdown("""
# #     <style>
# #         .main {
# #             background-color: #f5f5f5;
# #         }
# #         h1 {
# #             color: #18E1D9;
# #             font-size: 32px;
# #         }
# #         .stButton > button {
# #             background-color: #0B0B15;
# #             color: white;
# #             font-size: 16px;
# #             border-radius: 5px;
# #             padding: 10px;
# #         }
# #         .stButton > button:hover {
# #             background-color: #18E1D9;
# #         }
# #         .stNumberInput input {
# #             border: 2px solid #18E1D9;
# #             border-radius: 8px;
# #             font-size: 16px;
# #         }
# #         .stTextInput input {
# #             border: 2px solid #18E1D9;
# #             border-radius: 8px;
# #             font-size: 16px;
# #         }
# #         .stWrite {
# #             color: #0B0B15;
# #             font-size: 18px;
# #         }
# #     </style>
# # """, unsafe_allow_html=True)

# # st.write("Please enter the data to make a prediction:")

# # # Create input fields for each feature
# # رأس_المال = st.number_input('رأس المال',  format="%.3f")
# # نسبة_السيولة = st.number_input('نسبة السيولة', format="%.3f")
# # مضاعف_حقوق_الملكية = st.number_input('مضاعف حقوق الملكية',  format="%.3f")
# # صافي_الربح = st.number_input('صافي الربح',  format="%.3f")
# # معدل_دوران_الاصول = st.number_input('معدل دوران الاصول',  format="%.3f")
# # نسبة_الديون_الى_حقوق_الملكية = st.number_input('نسبة الديون الى حقوق الملكية',  format="%.3f")
# # الديون_الى_اجمالي_الخصوم = st.number_input('الديون الى اجمالي الخصوم',  format="%.3f")
# # العائد_على_الأصول_ROA = st.number_input('العائد على الأصول ROA',  format="%.3f")
# # العائد_على_حقوق_الملكية_ROI = st.number_input('العائد على حقوق الملكية ROI', format="%.3f")

# # # Collect all inputs into a DataFrame
# # input_data_manual = pd.DataFrame({
# #     'رأس المال': [رأس_المال],
# #     'نسبة السيولة': [نسبة_السيولة],
# #     'مضاعف حقوق الملكية': [مضاعف_حقوق_الملكية],
# #     'صافي الربح': [صافي_الربح],
# #     'معدل دوران الاصول': [معدل_دوران_الاصول],
# #     'نسبة الديون الى حقوق الملكية': [نسبة_الديون_الى_حقوق_الملكية],
# #     'الديون الى اجمالي الخصوم': [الديون_الى_اجمالي_الخصوم],
# #     'العائد على الأصول ROA': [العائد_على_الأصول_ROA],
# #     'العائد على حقوق الملكية ROI': [العائد_على_حقوق_الملكية_ROI]  # تأكد من أن الاسم هنا صحيح
# # })


# # input_data_manual = input_data_manual[selected_features]

# # st.write(f"Input shape after selection: {input_data_manual.shape}")


# # # طباعة أسماء الميزات المستخدمة أثناء التدريب
# # print("Features used in training:", scaler.feature_names_in_)

# # # طباعة الأسماء في الدخل اليدوي للتحقق من التطابق
# # print("Features in input data:", input_data_manual.columns.tolist())

# # # ضمان استخدام نفس الميزات بنفس الترتيب
# # input_data_manual = input_data_manual[scaler.feature_names_in_]

# # # تحويل البيانات باستخدام نفس `scaler` و `selector`
# # input_data_scaled_manual = scaler.transform(input_data_manual)
# # input_data_selected_manual = selector.transform(input_data_scaled_manual)

# # print("Features used in training:", scaler.feature_names_in_)
# # print("Features in input data:", input_data_manual.columns.tolist())

# # # Add a prediction button
# # if st.button('Predict Classification'):
# #     # Make prediction
# #     prediction = model.predict(input_data_selected_manual)
# #     predicted_class = label_encoder.inverse_transform(prediction)[0]

# #     # Display the result with styling
# #     st.markdown(f"""
# #         <div style="background-color: #18E1D9; color: white; padding: 20px; border-radius: 10px; font-size: 24px; text-align: center;">
# #             <strong>Predicted Classification: {predicted_class}</strong>
# #         </div>
# #     """, unsafe_allow_html=True)


# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load the saved model and preprocessing objects
# model = joblib.load('xgb_model.pkl')
# selector = joblib.load('selector.pkl')
# scaler = joblib.load('scaler.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

# # Define the selected features
# selected_features = [
#     'العام المالي', 'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 'معدل دوران الاصول',
#     'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
# ]

# # UI Design
# st.title("AI Prediction System")
# st.markdown("""
#     <style>
#         .main { background-color: #f5f5f5; }
#         h1 { color: #18E1D9; font-size: 32px; }
#         .stButton > button {
#             background-color: #0B0B15; color: white; font-size: 16px;
#             border-radius: 5px; padding: 10px;
#         }
#         .stButton > button:hover { background-color: #18E1D9; }
#         .stNumberInput input, .stTextInput input {
#             border: 2px solid #18E1D9; border-radius: 8px; font-size: 16px;
#         }
#         .stWrite { color: #0B0B15; font-size: 18px; }
#     </style>
# """, unsafe_allow_html=True)

# st.write("Please enter the data to make a prediction:")

# # Create input fields for each feature
# user_inputs = {}
# for feature in selected_features:
#     user_inputs[feature] = st.number_input(feature, format="%.3f")

# # Convert inputs into a DataFrame
# input_data_manual = pd.DataFrame([user_inputs])

# # Ensure feature order matches the training data
# try:
#     input_data_manual = input_data_manual[scaler.feature_names_in_]
# except AttributeError:
#     st.error("Error: The scaler object does not have 'feature_names_in_'. Ensure the correct scaler is loaded.")
#     st.stop()

# # Standardize and transform data
# input_data_scaled_manual = scaler.transform(input_data_manual)
# input_data_selected_manual = selector.transform(input_data_scaled_manual)

# # Predict Classification
# if st.button('Predict Classification'):
#     try:
#         prediction = model.predict(input_data_selected_manual)
#         predicted_class = label_encoder.inverse_transform(prediction)[0]

#         # Display the result with styling
#         st.markdown(f"""
#             <div style="background-color: #18E1D9; color: white; padding: 20px; border-radius: 10px; font-size: 24px; text-align: center;">
#                 <strong>Predicted Classification: {predicted_class}</strong>
#             </div>
#         """, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Prediction Error: {e}")


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessing objects
model = joblib.load('xgb_model.pkl')
selector = joblib.load('selector.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the selected features
selected_features = [
    'العام المالي', 'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 'معدل دوران الاصول',
    'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
]

# UI Design
st.title("Credit Category Prediction")

st.write("Please enter the data to make a prediction:")

# Create input fields for each feature
user_inputs = {}
for feature in selected_features:
    user_inputs[feature] = st.number_input(feature, format="%.3f")

# Convert inputs into a DataFrame
input_data_manual = pd.DataFrame([user_inputs])

# Ensure feature order matches the training data
try:
    input_data_manual = input_data_manual[scaler.feature_names_in_]
except AttributeError:
    st.error("Error: The scaler object does not have 'feature_names_in_'. Ensure the correct scaler is loaded.")
    st.stop()

# Standardize and transform data
input_data_scaled_manual = scaler.transform(input_data_manual)
input_data_selected_manual = selector.transform(input_data_scaled_manual)

# Predict Classification
if st.button('Predict Classification'):
    try:
        prediction = model.predict(input_data_selected_manual)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        # Display the result
        st.markdown(f"""
            <div style="background-color: #18E1D9; color: white; padding: 20px; border-radius: 10px; font-size: 24px; text-align: center;">
                <strong>Predicted Classification: {predicted_class}</strong>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Show feature importance
if st.button("Show Feature Importances"):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': selected_features[:len(feature_importances)],
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    st.write("### Feature Importances")
    st.dataframe(importance_df)

# Show Encoding Mapping
if st.button("Show Encoding Mapping"):
    encoding_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    encoding_df = pd.DataFrame(list(encoding_dict.items()), columns=['Original Class', 'Encoded Value'])

    st.write("### Encoding Mapping")
    st.dataframe(encoding_df)
