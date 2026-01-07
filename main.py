import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model pipeline
@st.cache_resource
def load_model(filename='model_xgb.pkl'):
    """Load the pre-trained model pipeline."""
    try:
        pipeline = joblib.load(filename)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file '{filename}' not found.")
        st.error("Please run 'python model_train.py' first to train and save the model.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        st.stop()

model_pipeline = load_model()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="US Health Insurance Cost Predictor",
    layout="centered"
)

st.title("💸 US Health Insurance Cost Predictor")
st.markdown("Estimate your annual medical charges using a Machine Learning model.")

# --- User Input Fields (in a form for better structure) ---
with st.form("prediction_form"):
    st.header("Personal Details", divider='gray')

    # Row 1: Age and BMI
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=64, value=30)
    with col2:
        bmi = st.number_input("BMI (e.g., 25.0)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)

    # Row 2: Sex and Children
    col3, col4 = st.columns(2)
    with col3:
        sex = st.selectbox("Gender", ["Male", "Female"])  # 0 = Male, 1 = Female
        sex = 1 if sex == "Male" else 0  # Convert to 1 or 0
    with col4:
        children = st.slider("Number of Children", min_value=0, max_value=5, value=0)

    # Row 3: Smoker and Region
    col5, col6 = st.columns(2)
    with col5:
        smoker = st.selectbox("Smoker?", ["No", "Yes"])  # Yes/No options for smoker
        smoker = 1 if smoker == "Yes" else 0  # Convert to 1 or 0
    with col6:
        region = st.selectbox("Region", [ "Northeast", "Northwest", "Southeast", "Southwest"])  
        # Convert region to categorical value (0 = Northwest, 1 = Southeast, etc.)
        region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
        region = region_mapping[region]

    # Prediction Button
    submitted = st.form_submit_button("💰 Predict Annual Charges", type="primary")

# --- Prediction Logic ---
if submitted:
    # 1. Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # 2. Make Prediction using the pre-trained pipeline (which handles encoding)
    try:
        prediction = model_pipeline.predict(input_data)[0]

        # 3. Display Result
        st.subheader("Prediction Result")
        st.success(f"Estimated Annual Medical Charges: **${abs(prediction):,.2f}**")
        
        st.markdown("---")
        # Optional: Add insights based on key features
        if smoker == 1:  # Check for smoking status
            st.warning("⚠️ **Smoker status is the biggest driver of high insurance costs.**")
        
        if bmi >= 30:
            st.info("💡 High BMI often correlates with higher premiums.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
import streamlit as st
import pandas as pd

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("These project provide a holistic view of health-related insurance factors and help identify high-risk demographics, cost drivers, and geographical patterns.")

def load_dataset(path, sep=";"):
    df = pd.read_csv(path, sep=sep)

    st.title("US Average Insurance Fees")

    # ======================
    # Gender-based averages
    # ======================
    st.subheader("Average Insurance Fees by Gender")

    gender_avg = df.groupby("sex")["charges"].mean()
    st.write(f"👨 **Average fees for males:** ${gender_avg.loc['male']:,.2f}")
    st.write(f"👩 **Average fees for females:** ${gender_avg.loc['female']:,.2f}")

    # ======================
    # Smoker-based averages
    # ======================
    st.subheader("Average Insurance Fees by Smoking Status")

    smoker_avg = df.groupby("smoker")["charges"].mean()
    st.write(f"🚬 **Average fees for smokers:** ${smoker_avg.loc['yes']:,.2f}")
    st.write(f"🚭 **Average fees for non-smokers:** ${smoker_avg.loc['no']:,.2f}")

    # ======================
    # Region-based averages
    # ======================
    st.subheader("Insurance Fees per Region")

    region_avg = df.groupby("region")["charges"].mean()

    
    st.dataframe(region_avg.reset_index())
    st.bar_chart(region_avg)

    return df

from pathlib import Path

path = Path('US_insurance.csv', sep=';')
if not path.exists():
    st.error(f"File not found: {path.resolve()}")
else:
    df = pd.read_csv(path)






