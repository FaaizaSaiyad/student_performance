import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page config and style
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="wide", page_icon="🎯")

# Custom CSS for better fonts and colors
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
        color: #212121;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-weight: 700;
        color: #4B8BBE;
        font-size: 3rem;
    }
    .subtitle {
        color: #306998;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze, visualize, and predict student pass/fail outcomes.</p>', unsafe_allow_html=True)

# Load dataset with spinner
with st.spinner("Loading and preparing data..."):
    df = pd.read_csv('StudentsPerformance.csv')
    df['pass_fail'] = ((df['math score'] >= 40) & (df['reading score'] >= 40) & (df['writing score'] >= 40)).astype(int)

    # Encode categorical features
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

# Model Evaluation Card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar inputs for interactive prediction
st.sidebar.header("🔍 Predict Pass/Fail")
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

def predict_pass_fail(raw_input_dict):
    encoded_input = {}
    for key in raw_input_dict:
        if key in label_encoders:
            encoded_input[key] = label_encoders[key].transform([raw_input_dict[key]])[0]
        else:
            encoded_input[key] = raw_input_dict[key]
    input_df = pd.DataFrame([encoded_input])
    prediction = model.predict(input_df)
    return "✅ Pass" if prediction[0] == 1 else "❌ Fail"

# Prediction result card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧮 Prediction Result")
    st.write("Input Features:")
    st.json(input_data)
    result = predict_pass_fail(input_data)
    if result == "✅ Pass":
        st.success(f"Prediction: {result}")
    else:
        st.error(f"Prediction: {result}")
    st.markdown('</div>', unsafe_allow_html=True)

# Visualizations in columns
st.subheader("📊 Data Visualizations")
df_viz = pd.read_csv('StudentsPerformance.csv')
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

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Name — Powered by Streamlit & scikit-learn")
