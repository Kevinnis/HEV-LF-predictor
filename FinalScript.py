import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sksurv
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
    TT = (TT - 18.49978) /3.781903

    feature_values_alf = [INR, TBIL, AST, HDL, PLT, NEU, TT]
    features_alf = np.array([feature_values_alf])

    # Predict button for HEV-ALF
    if st.button("Predict HEV-ALF Risk"):
        risk_score = gbm_mod1.predict(features_alf)[0]
    # Get the cumulative hazard 
        hazard_functions = gbm_mod1.predict_cumulative_hazard_function(features_alf)

    # Calculate the alf probabilities at 7, 14, and 28 days
    alf_probabilities = []
    for hazard in hazard_functions:
        # Get the cumulative hazard at 7, 14, and 28 days
        hazard_7 = hazard(7)  # 7 days
        hazard_14 = hazard(14)  # 14 days
        hazard_28 = hazard(28)  # 28 days
        
        # Calculate the probability
        prob_7 = 1 - np.exp(-hazard_7)
        prob_14 = 1 - np.exp(-hazard_14)
        prob_28 = 1 - np.exp(-hazard_28)
        
        # Append the probabilities for this sample
        alf_probabilities.append([prob_7, prob_14, prob_28])
    
        # Convert to numpy array for easy handling and rounding
        alf_probabilities = np.array(alf_probabilities)

        st.markdown("<h3 style='font-weight: bold;'>Prediction Results</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.2f}</h3>", unsafe_allow_html=True)

        if risk_score >= 0.3670939:
            st.markdown(f"<h3 style='text-align: center; color: red;'>High Risk</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center; color: green;'>Low Risk</h3>", unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='text-align: center;'>7 day HEV-ALF probability: {alf_probabilities[0][0]*100:.2f}% </h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>14 day HEV-ALF probability: {alf_probabilities[0][1]*100:.2f}% </h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>28 day HEV-ALF probability: {alf_probabilities[0][2]*100:.2f}% </h3>", unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("<h3 style='font-weight: bold;'>Prediction Interpretations</h3>", unsafe_allow_html=True)
        shap_values = shap_explainer1(features_alf)
        features_df = pd.DataFrame([feature_values_alf], columns=["INR", "TBIL", "AST", "HDL", "PLT", "NEU", "TT"])
        shap.plots.force(shap_values.base_values, shap_values.values[0], features_df, matplotlib=True)
        plt.savefig("shap_force_plot_alf.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot_alf.png")

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
        risk_score = gbm_mod2.predict(features_aclf)[0]
        st.markdown("<h3 style='font-weight: bold;'>Prediction Results</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.2f}</h3>", unsafe_allow_html=True)
    # Get the cumulative hazard 
        hazard_functions = gbm_mod2.predict_cumulative_hazard_function(features_aclf)

    # Calculate the aclf probabilities at 7, 14, and 28 days
        aclf_probabilities = []
    for hazard in hazard_functions:
        # Get the cumulative hazard at 7, 14, and 28 days
        hazard_7 = hazard(7)  # 7 days
        hazard_14 = hazard(14)  # 14 days
        hazard_28 = hazard(28)  # 28 days
        
        # Calculate the probability
        prob_7 = 1 - np.exp(-hazard_7)
        prob_14 = 1 - np.exp(-hazard_14)
        prob_28 = 1 - np.exp(-hazard_28)
        
        # Append the probabilities for this sample
        aclf_probabilities.append([prob_7, prob_14, prob_28])
    
        # Convert to numpy array for easy handling and rounding
        aclf_probabilities = np.array(aclf_probabilities)
       
        if risk_score >= 0.462816: 
            st.markdown(f"<h3 style='text-align: center; color: red;'>High Risk</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center; color: green;'>Low Risk</h3>", unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='text-align: center;'>7 day HEV-ACLF probability: {aclf_probabilities[0][0]*100:.2f}% </h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>14 day HEV-ACLF probability: {aclf_probabilities[0][1]*100:.2f}% </h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>28 day HEV-ACLF probability: {aclf_probabilities[0][2]*100:.2f}% </h3>", unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("<h3 style='font-weight: bold;'>Prediction Interpretations</h3>", unsafe_allow_html=True)
        shap_values = shap_explainer2(features_aclf)
        features_df = pd.DataFrame([feature_values_aclf], columns=["INR", "TBIL", "Na", "HDL", "URA"])
        shap.plots.force(shap_values.base_values, shap_values.values[0], features_df, matplotlib=True)
        plt.savefig("shap_force_plot_aclf.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot_aclf.png")

# Footer
st.caption('Version: 20250221 [This is currently a demo version for review]')
st.caption('Contact: wangjienjmu@126.com')