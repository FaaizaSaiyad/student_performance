import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Student Performance App", layout="centered")

# Title
st.title("🎓 Student Performance Analysis & Prediction")

# Load dataset
df = pd.read_csv('StudentsPerformance.csv')

# Display raw data
with st.expander("📊 Show Raw Data"):
    st.dataframe(df.head())

# Create target column
df['pass_fail'] = ((df['math score'] >= 40) & (df['reading score'] >= 40) & (df['writing score'] >= 40)).astype(int)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and labels
X = df.drop(['math score', 'reading score', 'writing score', 'pass_fail'], axis=1)
y = df['pass_fail']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation
st.subheader("📈 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Prediction function
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

# Sample prediction
st.subheader("🔍 Sample Prediction")
sample = {
    'gender': 'female',
    'race/ethnicity': 'group B',
    'parental level of education': "bachelor's degree",
    'lunch': 'standard',
    'test preparation course': 'completed'
}
st.write("Input:", sample)
st.write("Prediction:", predict_pass_fail(sample))

# Visualizations
st.subheader("📊 Data Visualizations")

# Load raw data again for visualization
df_viz = pd.read_csv('StudentsPerformance.csv')
df_viz['pass_fail'] = ((df_viz['math score'] >= 40) &
                       (df_viz['reading score'] >= 40) &
                       (df_viz['writing score'] >= 40)).map({1: "Pass", 0: "Fail"})
df_viz['average_score'] = (df_viz['math score'] + df_viz['reading score'] + df_viz['writing score']) / 3

# Pass/Fail Count
fig1, ax1 = plt.subplots()
sns.countplot(data=df_viz, x='pass_fail', palette='Set2', ax=ax1)
ax1.set_title("Pass vs Fail Count")
st.pyplot(fig1)

# Gender-wise Pass/Fail
fig2, ax2 = plt.subplots()
sns.countplot(data=df_viz, x='gender', hue='pass_fail', palette='coolwarm', ax=ax2)
ax2.set_title("Gender-wise Pass/Fail")
st.pyplot(fig2)

# Parental Education vs Score
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.barplot(data=df_viz, x='parental level of education', y='average_score', palette='viridis', ax=ax3)
ax3.set_title("Parental Education vs Avg. Score")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
st.pyplot(fig3)
