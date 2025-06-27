# app.py
import streamlit as st
import joblib

# Load saved model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("📰 Fake News Classifier")

# Input box
text = st.text_area("Enter a news article text here:")

# Button to classify
if st.button("Check if it's Fake or Real"):
    if text.strip() == "":
        st.warning("Please enter some news content.")
    else:
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        if prediction[0] == 1:
            st.success("✅ This is Real News.")
        else:
            st.error("❌ This is Fake News.")
