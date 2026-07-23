import sys
!{sys.executable} -m pip install streamlit

%%writefile streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Assuming exampleModel.pkl is in the same directory
try:
    model = joblib.load('exampleModel.pkl')
except FileNotFoundError:
    st.error("Error: 'exampleModel.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

st.set_page_config(page_title="Health Condition Predictor", layout="centered")

st.title('Health Condition Predictor')
st.markdown("### Enter your health statistics to get a prediction")

# Define default values for all inputs
default_values = {
    'sleep_duration': 7.0,
    'heart_rate': 70,
    'bmi': 25.0,
    'calorie_expenditure': 2000.0,
    'steps': 7000,
    'blood_pressure_systolic': 120,
    'blood_pressure_diastolic': 80,
    'oldpeak': 1.0,
    'cholesterol': 200,
    'glucose': 100,
    'age': 30,
    'stress_level': 3,
    'gender': 'male'
}

# Initialize session state with default values if not already present
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

def reset_all_stats():
    for key, value in default_values.items():
        st.session_state[key] = value

# Helper function for individual resets
def reset_individual_stat(stat_key):
    st.session_state[stat_key] = default_values[stat_key]

with st.sidebar:
    st.header("User Input Features")

    if st.button('Reset All Stats', key='reset_all_button'):
        reset_all_stats()
        st.experimental_rerun() # Rerun to apply the reset values immediately

    st.markdown("---") # Separator

    # Input fields for all features (excluding 'id')
    # Numerical features - using number_input or sliders
    # Each input now gets a unique key and its value is managed by session_state
    # Also added individual reset buttons

    # Sleep Duration
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0.0, max_value=24.0, value=st.session_state.sleep_duration, step=0.1, key='input_sleep_duration')
    with col2:
        if st.button('Reset', key='reset_sleep_duration'):
            reset_individual_stat('sleep_duration')
            st.experimental_rerun()

    # Heart Rate
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.heart_rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200, value=st.session_state.heart_rate, key='input_heart_rate')
    with col2:
        if st.button('Reset', key='reset_heart_rate'):
            reset_individual_stat('heart_rate')
            st.experimental_rerun()

    # BMI
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=st.session_state.bmi, step=0.1, key='input_bmi')
    with col2:
        if st.button('Reset', key='reset_bmi'):
            reset_individual_stat('bmi')
            st.experimental_rerun()

    # Calorie Expenditure
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.calorie_expenditure = st.number_input('Calorie Expenditure (kcal)', min_value=0.0, max_value=5000.0, value=st.session_state.calorie_expenditure, step=10.0, key='input_calorie_expenditure')
    with col2:
        if st.button('Reset', key='reset_calorie_expenditure'):
            reset_individual_stat('calorie_expenditure')
            st.experimental_rerun()

    # Steps
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.steps = st.number_input('Steps', min_value=0, max_value=30000, value=st.session_state.steps, key='input_steps')
    with col2:
        if st.button('Reset', key='reset_steps'):
            reset_individual_stat('steps')
            st.experimental_rerun()

    # Blood Pressure Systolic
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.blood_pressure_systolic = st.number_input('Blood Pressure Systolic (mmHg)', min_value=50, max_value=250, value=st.session_state.blood_pressure_systolic, key='input_blood_pressure_systolic')
    with col2:
        if st.button('Reset', key='reset_blood_pressure_systolic'):
            reset_individual_stat('blood_pressure_systolic')
            st.experimental_rerun()

    # Blood Pressure Diastolic
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.blood_pressure_diastolic = st.number_input('Blood Pressure Diastolic (mmHg)', min_value=30, max_value=150, value=st.session_state.blood_pressure_diastolic, key='input_blood_pressure_diastolic')
    with col2:
        if st.button('Reset', key='reset_blood_pressure_diastolic'):
            reset_individual_stat('blood_pressure_diastolic')
            st.experimental_rerun()

    # Oldpeak
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.oldpeak = st.number_input('Oldpeak (ST depression induced by exercise relative to rest)', min_value=0.0, max_value=6.0, value=st.session_state.oldpeak, step=0.1, key='input_oldpeak')
    with col2:
        if st.button('Reset', key='reset_oldpeak'):
            reset_individual_stat('oldpeak')
            st.experimental_rerun()

    # Cholesterol
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=st.session_state.cholesterol, key='input_cholesterol')
    with col2:
        if st.button('Reset', key='reset_cholesterol'):
            reset_individual_stat('cholesterol')
            st.experimental_rerun()

    # Glucose
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=300, value=st.session_state.glucose, key='input_glucose')
    with col2:
        if st.button('Reset', key='reset_glucose'):
            reset_individual_stat('glucose')
            st.experimental_rerun()

    # Age
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.age = st.number_input('Age (years)', min_value=0, max_value=120, value=st.session_state.age, key='input_age')
    with col2:
        if st.button('Reset', key='reset_age'):
            reset_individual_stat('age')
            st.experimental_rerun()

    # Stress Level
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.stress_level = st.slider('Stress Level (1-5)', min_value=1, max_value=5, value=st.session_state.stress_level, key='input_stress_level')
    with col2:
        if st.button('Reset', key='reset_stress_level'):
            reset_individual_stat('stress_level')
            st.experimental_rerun()

    # Gender
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.session_state.gender = st.selectbox('Gender', ['male', 'female', 'other'], index=['male', 'female', 'other'].index(st.session_state.gender), key='input_gender')
    with col2:
        if st.button('Reset', key='reset_gender'):
            reset_individual_stat('gender')
            st.experimental_rerun()


# Create a DataFrame from the inputs
# Use st.session_state directly here as the values are updated by the widgets
input_data = pd.DataFrame([{
    'id': 0, # Placeholder id, not used for prediction but expected by preprocessor
    'sleep_duration': st.session_state.sleep_duration,
    'heart_rate': st.session_state.heart_rate,
    'bmi': st.session_state.bmi,
    'calorie_expenditure': st.session_state.calorie_expenditure,
    'steps': st.session_state.steps,
    'blood_pressure_systolic': st.session_state.blood_pressure_systolic,
    'blood_pressure_diastolic': st.session_state.blood_pressure_diastolic,
    'oldpeak': st.session_state.oldpeak,
    'cholesterol': st.session_state.cholesterol,
    'glucose': st.session_state.glucose,
    'gender': st.session_state.gender,
    'age': st.session_state.age,
    'stress_level': st.session_state.stress_level
}])

# Make prediction
if st.button('Predict Health Condition'):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader('Prediction Result:')
        st.success(f"The predicted health condition is: **{prediction[0]}**")

        st.subheader('Prediction Probabilities:')
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.dataframe(proba_df.T.rename(columns={0: 'Probability'}))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")