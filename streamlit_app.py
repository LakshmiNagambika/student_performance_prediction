import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🎓 Student Performance Prediction System")

# Load dataset
data = pd.read_csv("student_performance_dataset.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# Features and target
X = data[["study_hours","attendance","previous_marks","assignment_score"]]
y = data["final_marks"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

st.header("Enter Student Details")

study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0)
previous_marks = st.number_input("Previous Marks", min_value=0.0)
assignment_score = st.number_input("Assignment Score", min_value=0.0)

if st.button("Predict Performance"):
    
    new_student = np.array([[study_hours,attendance,previous_marks,assignment_score]])
    
    predicted_marks = model.predict(new_student)[0]

    st.subheader(f"Predicted Marks: {round(predicted_marks,2)}")

    if predicted_marks >= 40:
        st.success("Result: PASS")
    else:
        st.error("Result: FAIL")