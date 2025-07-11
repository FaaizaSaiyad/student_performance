import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Cache dataset loading
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
    
    X = df.drop('pass_fail', axis=1)
    y = df['pass_fail']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model, label_encoders, df

# Load data and train model
df = load_data()
model, label_encoders, df = train_model(df)

st.title("🎓 Student Pass/Fail Predictor")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Pass vs Fail Count")
fig1, ax1 = plt.subplots()
sns.countplot(x=df['pass_fail'].map({1:'Pass', 0:'Fail'}), palette='Set2', ax=ax1)
ax1.set_xlabel("Result")
ax1.set_ylabel("Number of Students")
st.pyplot(fig1)

st.subheader("Gender vs Pass/Fail")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x='gender', hue=df['pass_fail'].map({1:'Pass', 0:'Fail'}), palette='coolwarm', ax=ax2)
ax2.set_xlabel("Gender")
ax2.set_ylabel("Number of Students")
ax2.legend(title="Result")
st.pyplot(fig2)

st.subheader("Parental Level of Education vs Average Score")
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.barplot(data=df, x='parental level of education', y='average_score', palette='viridis', ax=ax3)
ax3.set_xlabel("Parental Level of Education")
ax3.set_ylabel("Average Score")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig3)

st.markdown("---")
st.header("Predict Student Pass/Fail")

# User inputs
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
race = st.selectbox("Race/Ethnicity", label_encoders['race/ethnicity'].classes_)
parent_edu = st.selectbox("Parental Level of Education", label_encoders['parental level of education'].classes_)
lunch = st.selectbox("Lunch", label_encoders['lunch'].classes_)
prep_course = st.selectbox("Test Preparation Course", label_encoders['test preparation course'].classes_)

math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

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
            encoded[key] = label_encoders[key].transform([val])[0]
        else:
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
