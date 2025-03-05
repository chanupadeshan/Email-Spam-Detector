import joblib
import streamlit as st
import time

# Load the trained models and transformers
model = joblib.load('app/spam_model.pkl')  
cv = joblib.load('app/count_vectorizer.pkl')
standard = joblib.load('app/standard_scaler.pkl')

# Global variable for prediction probability
prediction_proba = 0

# Function to predict spam or not spam
def predict(email):
    email_count = cv.transform([email])
    standard_email = standard.transform(email_count)
    predict = model.predict(standard_email)
    
    global prediction_proba
    prediction_proba = model.predict_proba(standard_email)
    
    return "spam" if predict == 1 else "not spam"

def main():
    st.markdown("""
        <style>
            .stApp {
                background-color: #f0f8ff;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #008080;'>Email Spam Detection</h1>", unsafe_allow_html=True)
    
    email = st.text_area("Enter your email here", height=150)

    with st.spinner("Checking..."):
        time.sleep(1) 

    if st.button("Check spam"):
        if email:
            result = predict(email)

            if result == "spam":
                # Display the output
                st.markdown(f"""
                    <div style="background-color:#ff6347; padding:20px; border-radius:10px; color:white; text-align:center;">
                        <img src="https://img.icons8.com/ios-filled/50/000000/spam.png" alt="spam" style="vertical-align:middle;">
                        <h3>{result.capitalize()}</h3>
                        <p>Confidence: {prediction_proba[0, 1] * 100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
               
                st.markdown(f"""
                    <div style="background-color:#32cd32; padding:20px; border-radius:10px; color:white; text-align:center;">
                        <img src="https://img.icons8.com/ios-filled/50/000000/checked.png" alt="not spam" style="vertical-align:middle;">
                        <h3>{result.capitalize()}</h3>
                        <p>Confidence: {prediction_proba[0, 0] * 100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter an email to check.")


if __name__ == "__main__":
    main()
