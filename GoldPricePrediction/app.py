import streamlit as st
import plotly.express as px
import numpy as np
import joblib
import pandas as pd

# ===================== MODEL SETUP =====================
model_paths = {
    "Linear Regression": "Regression_model.pkl",
    "Ridge Regression": "best_ridge_model.pkl",
    "Random Forest": "best_random_forest_model.pkl"
}
scaler_path = "scaler.pkl"

# ===================== PREDICTION FUNCTION =====================
def predict_gold_rate(usd_inr_value, model_choice):
    if usd_inr_value <= 0:
        return 0.0, None, "⚠️ USD/INR exchange rate must be greater than 0."
    
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_paths[model_choice])

        usd_inr_scaled = scaler.transform(np.array([[usd_inr_value]]))
        predicted_gold_rate = model.predict(usd_inr_scaled)
        predicted_value = round(float(predicted_gold_rate[0]), 2)

        # Generate trend data
        usd_inr_range = np.linspace(usd_inr_value * 0.95, usd_inr_value * 1.05, 50)
        usd_inr_range_scaled = scaler.transform(usd_inr_range.reshape(-1, 1))
        predictions_range = model.predict(usd_inr_range_scaled).flatten()

        data = pd.DataFrame({
            'USD/INR': usd_inr_range,
            'Predicted Gold Rate (INR)': predictions_range
        })

        fig = px.line(
            data,
            x='USD/INR',
            y='Predicted Gold Rate (INR)',
            title=f"📈 Gold Rate Prediction vs USD/INR ({model_choice})",
            labels={"USD/INR": "USD/INR Exchange Rate", "Predicted Gold Rate (INR)": "Gold Rate (INR)"}
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="gold"),
            title_font=dict(size=20, color="gold")
        )

        return predicted_value, fig, "✅ Prediction successful!"
    
    except FileNotFoundError:
        return 0.0, None, "❌ Model or scaler file not found."
    except Exception as e:
        return 0.0, None, f"❌ Error: {e}"

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Gold Price Prediction Dashboard 💰", layout="wide")

# Custom Gold Theme CSS
st.markdown("""
    <style>
    body {
        background-color: #0e0e0e;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background: linear-gradient(145deg, #1b1b1b, #0e0e0e);
        border-radius: 20px;
        box-shadow: 0 0 25px rgba(255,215,0,0.3);
        padding: 25px;
    }
    h1, h2, h3 {
        text-align: center;
        color: gold !important;
        text-shadow: 0 0 10px #ffd700;
    }
    .gold-box {
        background: linear-gradient(90deg, #2d2d2d, #1a1a1a);
        border: 1px solid gold;
        border-radius: 12px;
        padding: 12px;
        color: #ffd700;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    footer {
        text-align: center;
        color: gray;
        margin-top: 20px;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("<h1>🏆 Gold Price Prediction Dashboard 🥇</h1>", unsafe_allow_html=True)

# ===================== PROJECT OVERVIEW =====================
st.markdown("""
### 💡 **Project Overview**
This project predicts the **future gold price (in INR)** based on the **USD/INR exchange rate** using different **machine learning models**.  
Gold prices are influenced by currency strength, inflation, and market trends — so predicting them can help investors, traders, and analysts make better decisions.

#### 🧭 **Objective**
To build a machine learning-based dashboard that:
- Predicts gold prices using USD/INR exchange rate data  
- Allows comparison between different regression models  
- Visualizes how gold price changes when USD/INR fluctuates  
""")

# ===================== MODEL EXPLANATION SECTION =====================
with st.expander("🔍 **Model Explanation**", expanded=False):
    st.markdown("""
    **1️⃣ Linear Regression**  
    - A simple model that assumes a straight-line relationship between USD/INR and Gold Price.  
    - Formula: `Gold_Price = a * (USD/INR) + b`  
    - Pros: Easy to interpret  
    - Cons: May underperform for non-linear patterns.  

    **2️⃣ Ridge Regression**  
    - A variation of linear regression with regularization (L2 penalty).  
    - It reduces model overfitting by shrinking large coefficients.  
    - Pros: More stable than simple linear regression.  
    - Cons: Slightly less interpretable.  

    **3️⃣ Random Forest Regression**  
    - An ensemble model that combines multiple decision trees.  
    - Captures complex, non-linear relationships.  
    - Pros: Very accurate and handles noise well.  
    - Cons: Slower and less interpretable.  
    """)

# ===================== INPUT AND PREDICTION SECTION =====================
st.markdown("---")
st.markdown("<h2>🎯 Gold Price Prediction</h2>", unsafe_allow_html=True)

gold_side_img = "https://tse1.mm.bing.net/th/id/OIP.VJmQFqe-uy_tUfzO74fXGwHaE_?cb=12&rs=1&pid=ImgDetMain&o=7&rm=3"
gold_bar_img = "https://media.istockphoto.com/id/1206809736/photo/pile-of-gold-bars.jpg?s=612x612&w=0&k=20&c=LaCbpj3_6mSKhbTd_CsUeFwqleXJFCBv9AypmbUPAvs="

col1, col2 = st.columns([1, 2])

# Left info and image
with col1:
    st.image(gold_side_img, use_container_width=True)
    st.markdown("""
        <p style="font-size:14px; color:#ccc;">
        Use the controls to predict the gold price based on current USD/INR value.  
        You can experiment with different models to see how their predictions vary.
        </p>
    """, unsafe_allow_html=True)

# Right: input controls and outputs
with col2:
    usd_inr_value = st.slider("💰 USD/INR Exchange Rate", 70.0, 100.0, 83.0, 0.1)
    model_choice = st.selectbox("🧠 Select Model", list(model_paths.keys()))

    if st.button("🔮 Predict Gold Price"):
        predicted_rate, fig, msg = predict_gold_rate(usd_inr_value, model_choice)

        if fig is not None:
            st.success(f"💵 Predicted Gold Rate: ₹{predicted_rate}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(msg)

    st.image(gold_bar_img, width=250)
    st.markdown('<div class="gold-box">💵 Estimated Gold Price Updates Below</div>', unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<footer>
Developed ❤️ by <b>Govind Narenders</b><br>
<a href='https://github.com/itzdineshx/Gold-price-prediction' target='_blank' style='color:gold;'>GitHub</a> |
<a href='https://www.linkedin.com/in/dinesh-x/' target='_blank' style='color:gold;'>LinkedIn</a>
</footer>
""", unsafe_allow_html=True)
