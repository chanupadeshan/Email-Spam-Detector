import joblib
import streamlit as st

model = joblib.load('spam_model.pkl')  
cv = joblib.load('count_vectorizer.pkl')


def predict(email):
    email_count = cv.transform([email])
    predict = model.predict(email_count)
    return "spam" if predict == 1 else "not spam"

def main():
    st.title("Email Spam Detection")
    email = st.text_area("Enter your email here")
    if st.button("Check spam"):
        if email:
            result = predict(email)
            st.success(f"Email is {result}")
        else:
            st.warning("Please enter email to check")
        
if __name__ == "__main__":
    main()