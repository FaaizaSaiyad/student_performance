import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page config & custom styles
st.set_page_config(page_title="🎓 Student Pass/Fail Predictor", layout="centered", page_icon="🎯")

st.markdown("""
<style>
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #4B8BBE;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #306998;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .card {
        background: #fff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgb(0 0 0 / 0.1);
        margin-bottom: 2rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🎓 Student Pass/Fail Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter student info to predict whether they will pass or fail.</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('StudentsPerformance.csv')
    return data

df_raw = load_data()

# Prepare data & model (cached)
@st.cache_data(show_spinner=False)
def prepare_model(df):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, label_encoders, acc, report

model, label_encoders, acc, report = prepare_model(df_raw)

# Sidebar inputs for prediction
st.sidebar.header("👤 Enter Student Details")

gender = st.sidebar.selectbox("Gender", label_encoders['gender'].classes_)
race = st.sidebar.selectbox("Race/Ethnicity", label_encoders['race/ethnicity'].classes_)
parent_edu = st.sidebar.selectbox("Parental Level of Education", label_encoders['parental level of education'].classes_)
lunch = st.sidebar.selectbox("Lunch", label_encoders['lunch'].classes_)
prep_course = st.sidebar.selectbox("Test Preparation Course", label_encoders['test preparation course'].classes_)

input_data = {
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parent_edu,
    'lunch': lunch,
    'test preparation course': prep_course
}

def predict_pass_fail(input_dict):
    encoded = {}
    for key in input_dict:
        if key in label_encoders:
            encoded[key] = label_encoders[key].transform([input_dict[key]])[0]
        else:
            encoded[key] = input_dict[key]
    df_input = pd.DataFrame([encoded])
    pred = model.predict(df_input)[0]
    return "✅ Pass" if pred == 1 else "❌ Fail"

# Show prediction nicely
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧮 Prediction Result")
st.write("**Input Student Details:**")
st.json(input_data)

result = predict_pass_fail(input_data)
if result == "✅ Pass":
    st.success(f"Prediction: {result} — This student is likely to pass! 🎉")
else:
    st.error(f"Prediction: {result} — This student may need extra support. 🙁")

st.markdown('</div>', unsafe_allow_html=True)

# Show model accuracy and report
with st.expander("📈 Model Accuracy and Classification Report"):
    st.write(f"**Model Accuracy:** {acc:.4f}")
    st.text(report)

# Data Visualizations
st.subheader("📊 Data Insights")

df_viz = df_raw.copy()
df_viz['pass_fail'] = ((df_viz['math score'] >= 40) & 
                      (df_viz['reading score'] >= 40) & 
                      (df_viz['writing score'] >= 40)).map({1: "Pass", 0: "Fail"})
df_viz['average_score'] = (df_viz['math score'] + df_viz['reading score'] + df_viz['writing score']) / 3

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_viz, x='pass_fail', palette='Set2', ax=ax1)
    ax1.set_title("Pass vs Fail Count")
    st.pyplot(fig1)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_viz, x='parental level of education', y='average_score', palette='viridis', ax=ax3)
    ax3.set_title("Parental Education vs Avg. Score")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig3)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_viz, x='gender', hue='pass_fail', palette='coolwarm', ax=ax2)
    ax2.set_title("Gender-wise Pass/Fail")
    st.pyplot(fig2)

st.markdown("---")
st.markdown("Made with ❤️ by Faaiza Saiyad | Powered by Streamlit & scikit-learn")

