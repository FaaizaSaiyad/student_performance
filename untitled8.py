import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Cache dataset loading
@st.cache_data
def load_data():
    return pd.read_csv('StudentsPerformance.csv')

# Cache model training
@st.cache_data(show_spinner=False)
def train_model(df):
    df = df.copy()
    # Target column: 1=Pass, 0=Fail based on scores >=40
    df['pass_fail'] = ((df['math score'] >= 40) & 
                       (df['reading score'] >= 40) & 
                       (df['writing score'] >= 40)).astype(int)
    
    # Label encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Features include all columns except pass_fail target
    X = df.drop('pass_fail', axis=1)
    y = df['pass_fail']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model, label_encoders

# Load data and train model
df = load_data()
model, label_encoders = train_model(df)

st.title("🎓 Student Pass/Fail Predictor")

st.write("Fill in the student details below and click **Predict** to see if the student is likely to pass or fail.")

# User input widgets
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
race = st.selectbox("Race/Ethnicity", label_encoders['race/ethnicity'].classes_)
parent_edu = st.selectbox("Parental Level of Education", label_encoders['parental level of education'].classes_)
lunch = st.selectbox("Lunch", label_encoders['lunch'].classes_)
prep_course = st.selectbox("Test Preparation Course", label_encoders['test preparation course'].classes_)

math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

# Prepare input dict for prediction
input_data = {
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parent_edu,
    'lunch': lunch,
    'test preparation course': prep_course,
    'math score': math_score,
    'reading score': reading_score,
    'writing score': writing_score
}

def predict(input_dict):
    encoded = {}
    for key, val in input_dict.items():
        if key in label_encoders:
            # encode categorical
            encoded[key] = label_encoders[key].transform([val])[0]
        else:
            # numeric features as is
            encoded[key] = val
    df_input = pd.DataFrame([encoded])
    pred = model.predict(df_input)[0]
    return "✅ Pass" if pred == 1 else "❌ Fail"

if st.button("Predict"):
    prediction = predict(input_data)
    if prediction == "✅ Pass":
        st.success(f"Prediction: {prediction} — This student is likely to pass! 🎉")
        st.balloons()
    else:
        st.error(f"Prediction: {prediction} — This student may need extra support. 🙁")
