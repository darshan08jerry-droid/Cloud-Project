import streamlit as st
from predict import predict_spam

# Configure app layout and appearance
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for premium styling
st.markdown('''
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #6B7280;
            text-align: center;
            margin-bottom: 2rem;
        }
        .spam-result {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 1rem;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            border: 1px solid #F87171;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .ham-result {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 1rem;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            border: 1px solid #34D399;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            width: 100%;
            background-color: #3B82F6;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #2563EB;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
    </style>
''', unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🛡️ Email Spam Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Machine Learning Classification</div>', unsafe_allow_html=True)
    
    st.write("---")
    
    # Input Area
    input_text = st.text_area("Enter your email or message content below:", height=200, placeholder="Type or paste the message here...")
    
    # Prediction Button
    if st.button("Check Spam"):
        if input_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing message..."):
                prediction, confidence = predict_spam(input_text)
                
            st.write("### Result:")
            if prediction == "Spam":
                st.markdown(f'<div class="spam-result">⚠️ SPAM DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ham-result">✅ NORMAL MESSAGE (NOT SPAM)</div>', unsafe_allow_html=True)
            
            # Display confidence
            st.write(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            
            # Progress bar based on confidence
            if prediction == "Spam":
                st.progress(float(confidence), text="Spam Probability")
            else:
                st.progress(float(confidence), text="Legitimate Probability")

if __name__ == "__main__":
    main()
