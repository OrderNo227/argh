import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Assuming exampleModel.pkl is in the same directory

st.set_page_config(page_title="Health Condition Predictor", layout="centered")

st.title('Health Condition Predictor')
st.markdown("### Enter your health statistics to get a prediction")

DEFAULTS = {
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
    'gender': 'male',
}


def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# Input fields for all features (excluding 'id')
with st.sidebar:
    st.header("User Input Features")

    if st.button("Reset all values", use_container_width=True):
        reset_inputs()

    # Numerical features - using number_input or sliders
    st.session_state.setdefault('sleep_duration', DEFAULTS['sleep_duration'])
    sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0.0, max_value=24.0,
                                     value=st.session_state['sleep_duration'], step=0.1, key='sleep_duration')
    st.session_state.setdefault('heart_rate', DEFAULTS['heart_rate'])
    heart_rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200,
                                 value=st.session_state['heart_rate'], key='heart_rate')
    st.session_state.setdefault('bmi', DEFAULTS['bmi'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=st.session_state['bmi'], step=0.1,
                          key='bmi')
    st.session_state.setdefault('calorie_expenditure', DEFAULTS['calorie_expenditure'])
    calorie_expenditure = st.number_input('Calorie Expenditure (kcal)', min_value=0.0, max_value=5000.0,
                                          value=st.session_state['calorie_expenditure'], step=10.0,
                                          key='calorie_expenditure')
    st.session_state.setdefault('steps', DEFAULTS['steps'])
    steps = st.number_input('Steps', min_value=0, max_value=30000, value=st.session_state['steps'], key='steps')
    st.session_state.setdefault('blood_pressure_systolic', DEFAULTS['blood_pressure_systolic'])
    blood_pressure_systolic = st.number_input('Blood Pressure Systolic (mmHg)', min_value=50, max_value=250,
                                              value=st.session_state['blood_pressure_systolic'], key='blood_pressure_systolic')
    st.session_state.setdefault('blood_pressure_diastolic', DEFAULTS['blood_pressure_diastolic'])
    blood_pressure_diastolic = st.number_input('Blood Pressure Diastolic (mmHg)', min_value=30, max_value=150,
                                               value=st.session_state['blood_pressure_diastolic'], key='blood_pressure_diastolic')
    st.session_state.setdefault('oldpeak', DEFAULTS['oldpeak'])
    oldpeak = st.number_input('Oldpeak (ST depression induced by exercise relative to rest)', min_value=0.0,
                              max_value=6.0, value=st.session_state['oldpeak'], step=0.1, key='oldpeak')
    st.session_state.setdefault('cholesterol', DEFAULTS['cholesterol'])
    cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400,
                                  value=st.session_state['cholesterol'], key='cholesterol')
    st.session_state.setdefault('glucose', DEFAULTS['glucose'])
    glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=300, value=st.session_state['glucose'],
                              key='glucose')
    st.session_state.setdefault('age', DEFAULTS['age'])
    age = st.number_input('Age (years)', min_value=0, max_value=120, value=st.session_state['age'], key='age')
    st.session_state.setdefault('stress_level', DEFAULTS['stress_level'])
    stress_level = st.slider('Stress Level (1-5)', min_value=1, max_value=5, value=st.session_state['stress_level'],
                             key='stress_level')

    # Categorical features
    st.session_state.setdefault('gender', DEFAULTS['gender'])
    gender_options = ['male', 'female', 'other']
    gender = st.selectbox('Gender', gender_options, index=gender_options.index(st.session_state['gender']),
                          key='gender')

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'id': 0,  # Placeholder id, not used for prediction but expected by preprocessor
    'sleep_duration': sleep_duration,
    'heart_rate': heart_rate,
    'bmi': bmi,
    'calorie_expenditure': calorie_expenditure,
    'steps': steps,
    'blood_pressure_systolic': blood_pressure_systolic,
    'blood_pressure_diastolic': blood_pressure_diastolic,
    'oldpeak': oldpeak,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'gender': gender,
    'age': age,
    'stress_level': stress_level
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
