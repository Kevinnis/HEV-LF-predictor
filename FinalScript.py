import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load models and explainers
gbm_mod1 = joblib.load("gbm_mod1.pkl")
shap_explainer1 = joblib.load("shap_explainer1.pkl")

gbm_mod2 = joblib.load("gbm_mod2.pkl")
shap_explainer2 = joblib.load("shap_explainer2.pkl")

# Set page configuration
st.set_page_config(
    page_title="HEV Risk Predictor",
    page_icon=":microbe:",
    layout="centered",
)

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["HEV-ALF Predictor", "HEV-ACLF Predictor"])

# Custom CSS for styling
st.markdown("""
    <style>
        .stNumberInput {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        div.stButton > button:first-child {
            display: block;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Function to calculate probabilities from hazard functions
def calculate_probabilities(hazard_functions):
    probabilities = []
    for hazard in hazard_functions:
        hazard_7 = hazard(7)  # 7 days
        hazard_14 = hazard(14)  # 14 days
        hazard_28 = hazard(28)  # 28 days
        prob_7 = 1 - np.exp(-hazard_7)
        prob_14 = 1 - np.exp(-hazard_14)
        prob_28 = 1 - np.exp(-hazard_28)
        probabilities.append([prob_7, prob_14, prob_28])
    return np.array(probabilities)

# Function to display prediction results
def display_results(risk_score, probabilities, risk_threshold, condition_name):
    st.markdown("<h3 style='font-weight: bold;'>Prediction Results</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.2f}</h3>", unsafe_allow_html=True)

    if risk_score >= risk_threshold:
        st.markdown(f"<h3 style='text-align: center; color: red;'>High Risk</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: green;'>Low Risk</h3>", unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='text-align: center;'>7 day {condition_name} probability: {probabilities[0][0]*100:.2f}% </h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>14 day {condition_name} probability: {probabilities[0][1]*100:.2f}% </h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>28 day {condition_name} probability: {probabilities[0][2]*100:.2f}% </h3>", unsafe_allow_html=True)

# Function to generate SHAP explanation plot
def generate_shap_plot(explainer, features, feature_names, output_file):
    shap_values = explainer(features)
    features_df = pd.DataFrame([features], columns=feature_names)
    shap.plots.force(shap_values.base_values, shap_values.values[0], features_df, matplotlib=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=1200)
    st.image(output_file)

# HEV-ALF Predictor Page
if page == "HEV-ALF Predictor":
    st.title("HEV-ALF Risk Predictor")
    st.caption('This online tool predicts the risk of hepatitis E virus-related acute liver failure (HEV-ALF) among hospitalized patients with acute hepatitis E.')

    # Input form for HEV-ALF
    st.markdown("### Input the baseline features of the patient")
    INR = st.number_input("International normalized ratio (INR)", min_value=0.0, max_value=100.0, format="%.2f", key="INR_ALF")
    TBIL = st.number_input("Total bilirubin (TBIL) (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="TBIL_ALF")
    AST = st.number_input("Aspartate aminotransferase (AST) (U/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="AST_ALF")
    HDL = st.number_input("High-density lipoprotein-cholesterol (HDL-C) (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="HDL_ALF")
    PLT = st.number_input("Platelet count (PLT) (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="PLT_ALF")
    NEU = st.number_input("Neutrophil count (NEU) (10^9/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="NEU_ALF")
    TT = st.number_input("Thrombin time (TT) (s)", min_value=0.0, max_value=10000.0, format="%.2f", key="TT_ALF")

    # Z-score transformation (standardization) for HEV-ALF
    INR = (INR - 1.182121) / 0.3839814
    TBIL = (TBIL - 120.4193) / 113.3215
    AST = (AST - 517.3548) / 817.2993  
    HDL = (HDL - 0.7632815) / 0.4190331
    PLT = (PLT - 180.4892) / 69.47647
    NEU = (NEU - 4.023269) / 2.824536
    TT = (TT - 18.49978) / 3.781903

    feature_values_alf = [INR, TBIL, AST, HDL, PLT, NEU, TT]
    features_alf = np.array([feature_values_alf])

    # Predict button for HEV-ALF
    if st.button("Predict HEV-ALF Risk"):
        try:
            # Predict risk score
            risk_score = gbm_mod1.predict(features_alf)[0]

            # Get the cumulative hazard function
            hazard_functions = gbm_mod1.predict_cumulative_hazard_function(features_alf)

            # Calculate probabilities
            alf_probabilities = calculate_probabilities(hazard_functions)

            # Display results
            display_results(risk_score, alf_probabilities, 0.3670939, "HEV-ALF")

            # SHAP explanation
            st.markdown("<h3 style='font-weight: bold;'>Prediction Interpretations</h3>", unsafe_allow_html=True)
            generate_shap_plot(shap_explainer1, features_alf, ["INR", "TBIL", "AST", "HDL", "PLT", "NEU", "TT"], "shap_force_plot_alf.png")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# HEV-ACLF Predictor Page
elif page == "HEV-ACLF Predictor":
    st.title("HEV-ACLF Risk Predictor")
    st.caption('This online tool predicts the risk of hepatitis E virus-related acute-on-chronic liver failure (HEV-ACLF) among hospitalized patients with acute hepatitis E.')

    # Input form for HEV-ACLF
    st.markdown("### Input the baseline features of the patient")
    INR = st.number_input("International normalized ratio (INR)", min_value=0.0, max_value=100.0, format="%.2f", key="INR_ACLF")
    TBIL = st.number_input("Total bilirubin (TBIL) (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="TBIL_ACLF")
    Na = st.number_input("Sodium (Na) (mmol/L)", min_value=0.0, max_value=1000.0, format="%.2f", key="Na_ACLF")
    HDL = st.number_input("High-density lipoprotein-cholesterol (HDL-C) (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="HDL_ACLF")
    URA = st.number_input("Uric acid (URA) (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="URA_ACLF")

    # Z-score transformation (standardization) for HEV-ACLF
    INR = (INR - 1.244769) / 0.362144
    TBIL = (TBIL - 127.8267) / 123.9332
    Na = (Na - 138.7888) / 3.736281
    HDL = (HDL - 0.7207303) / 0.4138742
    URA = (URA - 279.6014) / 118.7426

    feature_values_aclf = [INR, TBIL, Na, HDL, URA]
    features_aclf = np.array([feature_values_aclf])

    # Predict button for HEV-ACLF
    if st.button("Predict HEV-ACLF Risk"):
        try:
            # Predict risk score
            risk_score = gbm_mod2.predict(features_aclf)[0]

            # Get the cumulative hazard function
            hazard_functions = gbm_mod2.predict_cumulative_hazard_function(features_aclf)

            # Calculate probabilities
            aclf_probabilities = calculate_probabilities(hazard_functions)

            # Display results
            display_results(risk_score, aclf_probabilities, 0.462816, "HEV-ACLF")

            # SHAP explanation
            st.markdown("<h3 style='font-weight: bold;'>Prediction Interpretations</h3>", unsafe_allow_html=True)
            generate_shap_plot(shap_explainer2, features_aclf, ["INR", "TBIL", "Na", "HDL", "URA"], "shap_force_plot_aclf.png")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.caption('Version: 20250221 [This is currently a demo version for review]')
st.caption('Contact: wangjienjmu@126.com')