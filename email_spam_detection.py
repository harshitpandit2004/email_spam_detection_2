import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("email_spam_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Email Spam Detection")

st.write("Enter the email details below to predict whether the email is Spam or Not Spam.")

email_length = st.number_input("Email Length", min_value=0, value=100)
num_special_chars = st.number_input("Number of Special Characters", min_value=-1, value=5)
num_links = st.number_input("Number of Links", min_value=0, value=1)


if st.button("Predict"):

    
    new_email = pd.DataFrame({
        "Email_Length": [email_length],
        "Num_Special_Chars": [num_special_chars],
        "Num_Links": [num_links]
    })


    if new_email.loc[0, "Email_Length"] <= 0:
        new_email.loc[0, "Email_Length"] = 100

    if new_email.loc[0, "Num_Special_Chars"] == -1:
        new_email.loc[0, "Num_Special_Chars"] = 0

    if new_email.loc[0, "Num_Links"] == 99:
        new_email.loc[0, "Num_Links"] = new_email["Num_Links"].median()

    

    new_email_scaled = scaler.transform(new_email)


    prediction = model.predict(new_email_scaled)

    if prediction[0] == 1:
        st.success("Prediction: Spam Email")
    else:
        st.success("Prediction: Not Spam")
