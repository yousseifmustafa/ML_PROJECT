import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("ملفات الموديل (pkl) مش موجودة! تأكد إنك عملت Run لآخر سيل في النوت بوك ونقلت الملفات هنا.")
        return None, None

model, scaler = load_assets()

EDUCATION_MAPPING = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4
}

NUMERIC_COLS = ['ChldNo', 'inc', 'famsize', 'AGE_YEARS', 'EMPLOYMENT_YEARS']

st.title("💳 Credit Card Approval Prediction System")
st.markdown("""
Enter the applicant's details below to predict their credit risk status.
**(Good Client vs. Bad Client)**
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
    education = st.selectbox("Education Level", list(EDUCATION_MAPPING.keys()))
    marital_status = st.selectbox("Marital Status", ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])
    
with col2:
    st.subheader("Financial Info")
    income = st.number_input("Annual Income (Total)", min_value=0.0, value=50000.0)
    income_type = st.selectbox("Income Type", ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])
    occupation = st.selectbox("Occupation", ['Unspecified', 'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
                                           'High skill tech staff', 'Accountants', 'Medicine staff', 'Cooking staff',
                                           'Security staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers',
                                           'Secretaries', 'Waiters/barmen staff', 'HR staff', 'IT staff', 'Realty agents'])
    employment_years = st.number_input("Years Employed", min_value=0.0, max_value=60.0, value=5.0)

with col3:
    st.subheader("Assets & Family")
    has_car = st.radio("Owns a Car?", ["Yes", "No"])
    has_realty = st.radio("Owns Real Estate?", ["Yes", "No"])
    children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    family_size = st.number_input("Family Size", min_value=1, max_value=20, value=2)
    housing_type = st.selectbox("Housing Type", ['House / apartment', 'With parents', 'Municipal apartment', 
                                               'Rented apartment', 'Office apartment', 'Co-op apartment'])

st.divider()
st.subheader("Contact Info")
c1, c2, c3 = st.columns(3)
with c1:
    work_phone = st.checkbox("Has Work Phone?")
with c2:
    phone = st.checkbox("Has Phone?")
with c3:
    email = st.checkbox("Has Email?")


def preprocess_input():
    input_data = {
        'Gender': 1 if gender == 'M' else 0, 
        'Car': 1 if has_car == 'Yes' else 0, 
        'Reality': 1 if has_realty == 'Yes' else 0, 
        'ChldNo': children,
        'inc': income,
        'famsize': family_size,
        'wkphone': 1 if work_phone else 0,
        'phone': 1 if phone else 0,
        'email': 1 if email else 0,
        'AGE_YEARS': age,
        'EMPLOYMENT_YEARS': employment_years,
        'IS_UNEMPLOYED': 1 if employment_years == 0 else 0,
        'edutp_encoded': EDUCATION_MAPPING[education]
    }
    
    df = pd.DataFrame([input_data])
    
    nominal_columns_values = {
        'inctp': income_type,
        'famtp': marital_status,
        'houtp': housing_type,
        'occyp': occupation
    }

    return df, nominal_columns_values

if st.button("🔮 Predict Risk Status", type="primary"):
    if model:
        base_df, nominal_vals = preprocess_input()
        
        try:
            model_features = model.feature_names_in_
        except:
            st.error("الموديل مش محتفظ بأسماء الـ Features. تأكد من حفظ الموديل صح.")
            st.stop()
            
        final_df = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
        
        for col in base_df.columns:
            if col in final_df.columns:
                final_df[col] = base_df[col]
        
        for prefix, val in nominal_vals.items():
            dummy_col = f"{prefix}_{val}"
            if dummy_col in final_df.columns:
                final_df[dummy_col] = 1.0
        
        final_df[NUMERIC_COLS] = scaler.transform(final_df[NUMERIC_COLS])
        
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1]
        
        st.subheader("Prediction Result:")
        if prediction == 0:
            st.success(f"✅ **Good Client (Approved)**")
            st.info(f"Risk Probability: {probability:.2%}")
        else:
            st.error(f"❌ **Bad Client (High Risk - Rejected)**")
            st.warning(f"Risk Probability: {probability:.2%}")
            
        with st.expander("Show Processed Data (For Debugging)"):
            st.dataframe(final_df)
    else:
        st.error("Model not loaded.")
