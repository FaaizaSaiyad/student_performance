import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load & prepare data + model (cache to speed up)
@st.cache_data
def load_data():
    return pd.read_csv('StudentsPerformance.csv')

@st.cache_data(show_spinner=False)
def train_model(df):
    df = df.copy()
    df['pass_fail'] = ((df['math score'] >= 40) & 
                       (df['reading score'] >= 40) & 
                       (df['writing score'] >= 40)).astype(int)
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df.drop(['math score', 'reading score', 'writing score', 'pass_fail'], axis=1)
    y = df['pass_fail']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model, label_encoders

df = load_data()
model, label_encoders = train_model(df)

st.title("🎓 Student Pass/Fail Predictor")

st.write("Enter student details and click **Predict** to check if the student is likely to pass or fail.")

# User inputs on main page
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
race = st.selectbox("Race/Ethnicity", label_encoders['race/ethnicity'].classes_)
parent_edu = st.selectbox("Parental Level of Education", label_encoders['parental level of education'].classes_)
lunch = st.selectbox("Lunch", label_encoders['lunch'].classes_)
prep_course = st.selectbox("Test Preparation Course", label_encoders['test preparation course'].classes_)

input_data = {
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parent_edu,
    'lunch': lunch,
    'test preparation course': prep_course
}

def predict(input_dict):
    encoded = {}
    for key in input_dict:
        encoded[key] = label_encoders[key].transform([input_dict[key]])[0]
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
