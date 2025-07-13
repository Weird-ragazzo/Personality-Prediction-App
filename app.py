import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# Configure page
st.set_page_config(
    page_title="🔮 Personality Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean and modern CSS with better contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        background-attachment: fixed;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem auto;
        max-width: 1200px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: none;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a202c;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 3px solid #3182ce;
        text-align: center;
    }
    
    .model-card {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-3px);
    }
    
    .model-card h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .model-card p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c5282 0%, #3182ce 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .result-card {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .result-card.extrovert {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
    }
    
    .result-card h2 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .result-card p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0;
    }
    
    .input-section {
        background: rgba(247, 250, 252, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #3182ce;
    }
    
    .input-section h4 {
        color: #1a202c;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .personality-info {
        background: rgba(247, 250, 252, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #38a169;
    }
    
    .personality-info h4 {
        color: #1a202c;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .personality-info p, .personality-info ul {
        color: #2d3748;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    .personality-info ul {
        padding-left: 1.2rem;
    }
    
    .personality-info li {
        margin: 0.3rem 0;
    }
    
    .model-selection {
        background: rgba(247, 250, 252, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e2e8f0;
    }
    
    .model-selection h4 {
        color: #1a202c;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .model-selection p {
        color: #4a5568;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .accuracy-badge {
        background: #38a169;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
        background: white;
        color: #1a202c;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3182ce;
        box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
    }
    
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 3rem;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
    }
    
    .footer p {
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Hide empty containers */
    .element-container:empty {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Load all trained models and their accuracies
@st.cache_resource
def load_models():
    try:
        models = {
            "Logistic Regression": {
                "model": joblib.load("PersonalityPredictionLogReg.pkl"),
                "accuracy": 0.99,
                "description": "A linear model that uses probabilities to make binary classifications. Perfect for understanding the relationship between features and personality types.",
                "icon": "📊"
            },
            "Decision Tree": {
                "model": joblib.load("PersonalityPredictionDT.pkl"),
                "accuracy": 0.99,
                "description": "A tree-structured model that creates decision rules based on feature thresholds. Great for interpretable decision-making processes.",
                "icon": "🌳"
            },
            "SVM": {
                "model": joblib.load("PersonalityPredictionSVM.pkl"),
                "accuracy": 0.99,
                "description": "Support Vector Machine finds the optimal boundary between personality types by maximizing the margin between classes.",
                "icon": "🎯"
            }
        }
        return models
    except Exception as e:
        st.error(f"⚠️ Model files not found. Please ensure these files are in the same directory:")
        st.error("• PersonalityPredictionLogReg.pkl")
        st.error("• PersonalityPredictionDT.pkl") 
        st.error("• PersonalityPredictionSVM.pkl")
        st.error(f"Error details: {str(e)}")
        st.stop()

models = load_models()

# Main content
st.markdown('''
<div class="main-container">
    <h1 class="main-title">🔮 Personality Prediction AI</h1>
    <p class="subtitle">Discover your personality type using advanced machine learning algorithms</p>
</div>
''', unsafe_allow_html=True)

# Model Performance Section
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="model-card">
        <h3>{models["Logistic Regression"]["icon"]} Logistic Regression</h3>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
            {models["Logistic Regression"]["accuracy"]*100:.0f}%
        </p>
        <p style="font-size: 0.9rem;">
            Linear probability-based classification
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="model-card">
        <h3>{models["Decision Tree"]["icon"]} Decision Tree</h3>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
            {models["Decision Tree"]["accuracy"]*100:.0f}%
        </p>
        <p style="font-size: 0.9rem;">
            Rule-based decision making
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="model-card">
        <h3>{models["SVM"]["icon"]} SVM</h3>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
            {models["SVM"]["accuracy"]*100:.0f}%
        </p>
        <p style="font-size: 0.9rem;">
            Maximum margin classification
        </p>
    </div>
    """, unsafe_allow_html=True)

# Model Selection Section
st.markdown('<div class="section-header">🔧 Choose Your AI Model</div>', unsafe_allow_html=True)

model_choice = st.selectbox(
    "Select a model for prediction:",
    list(models.keys()),
    help="Each model uses different algorithms to analyze your personality traits"
)

selected_model = models[model_choice]["model"]
model_accuracy = models[model_choice]["accuracy"]

st.markdown(f"""
<div class="model-selection">
    <h4>{models[model_choice]["icon"]} {model_choice} - Selected</h4>
    <p>{models[model_choice]["description"]}</p>
    <span class="accuracy-badge">Accuracy: {model_accuracy*100:.0f}%</span>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown('<div class="section-header">📝 Tell Us About Yourself</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="input-section">
        <h4>🏠 Personal Habits</h4>
    </div>
    """, unsafe_allow_html=True)
    
    time_spent_alone = st.number_input(
        "⏰ Time spent alone (hours per day)",
        min_value=0,
        max_value=24,
        value=8,
        step=1,
        help="How many hours do you typically spend alone each day?"
    )
    
    stage_fear = st.selectbox(
        "🎭 Do you have stage fear?",
        ["No", "Yes"],
        help="Are you comfortable speaking or performing in front of others?"
    )
    
    going_outside_count = st.number_input(
        "🚶 Times you go outside per week",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        help="How often do you leave your home in a typical week?"
    )
    
    going_outside_preference = st.selectbox(
        "🌞 Do you enjoy going outside?",
        ["Yes", "No"],
        help="Do you generally like outdoor activities and being outside?"
    )

with col2:
    st.markdown("""
    <div class="input-section">
        <h4>👥 Social Behavior</h4>
    </div>
    """, unsafe_allow_html=True)
    
    social_event_attendance = st.number_input(
        "🎉 Social events attended per month",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        help="How many social gatherings do you attend monthly?"
    )
    
    drained_after_socializing = st.selectbox(
        "😴 Feel drained after socializing?",
        ["No", "Yes"],
        help="Do you feel tired or need alone time after social interactions?"
    )
    
    friends_circle_size = st.number_input(
        "👥 Number of close friends",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        help="How many people do you consider close friends?"
    )
    
    post_frequency = st.number_input(
        "📱 Social media posts per week",
        min_value=0,
        max_value=100,
        value=3,
        step=1,
        help="How often do you post on social media platforms?"
    )

# Convert categorical inputs to numeric
stage_fear_num = 1 if stage_fear == "Yes" else 0
drained_after_socializing_num = 1 if drained_after_socializing == "Yes" else 0

# Prepare input for model prediction
input_data = np.array([[
    time_spent_alone,
    stage_fear_num,
    social_event_attendance,
    going_outside_count,
    drained_after_socializing_num,
    friends_circle_size,
    post_frequency
]])

# Session state for result
if 'result' not in st.session_state:
    st.session_state.result = None
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

# Prediction Section
st.markdown('<div class="section-header">🔮 Personality Prediction</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔮 Predict My Personality Type"):
        with st.spinner("🧠 AI is analyzing your personality..."):
            time.sleep(1)
            result = selected_model.predict(input_data)
            personality = "Extrovert" if result[0] == 1 else "Introvert"
            
            try:
                confidence = selected_model.predict_proba(input_data)[0].max() * 100
            except:
                confidence = model_accuracy * 100
            
            st.session_state.result = {
                "personality": personality,
                "model": model_choice,
                "confidence": confidence
            }
            st.session_state.prediction_count += 1
            st.balloons()

# Display result
if st.session_state.result:
    personality = st.session_state.result["personality"]
    model_used = st.session_state.result["model"]
    confidence = st.session_state.result["confidence"]
    
    personality_icon = "🌟" if personality == "Extrovert" else "🌙"
    result_class = "extrovert" if personality == "Extrovert" else ""
    
    st.markdown(f"""
    <div class="result-card {result_class}">
        <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">
            {personality_icon} You are an {personality}!
        </h2>
        <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
            Predicted using <strong>{model_used}</strong>
        </p>
        <p style="font-size: 1rem;">
            Confidence: {confidence:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)
    


    # Try Again Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Try Again with Different Inputs"):
            st.session_state.result = None
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="footer">
    <p><strong>Made with ❤️ by Dhruv Raghav</strong></p>
    <p>Powered by Machine Learning & Streamlit</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        Total Predictions: {st.session_state.prediction_count} | 
        Accuracy: 99% | 
        Models: 3 AI Algorithms
    </p>
</div>
""", unsafe_allow_html=True)