import gradio as gr
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
        return 0.0, None, "Error: USD/INR exchange rate must be greater than 0."
    
    try:
        scaler = joblib.load(scaler_path)
        model_path = model_paths.get(model_choice)
        model = joblib.load(model_path)

        usd_inr_scaled = scaler.transform(np.array([[usd_inr_value]]))
        predicted_gold_rate = model.predict(usd_inr_scaled)
        predicted_value = float(np.ravel(predicted_gold_rate)[0])
        predicted_value = round(predicted_value, 2)

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
            font=dict(color="gold")
        )

        return predicted_value, fig, "Prediction successful ✅"
    
    except FileNotFoundError:
        return 0.0, None, "Error: Model or scaler file not found."
    except Exception as e:
        return 0.0, None, f"Error: {str(e)}"


# ===================== BEAUTIFUL DASHBOARD UI =====================
gold_side_img = "https://tse1.mm.bing.net/th/id/OIP.VJmQFqe-uy_tUfzO74fXGwHaE_?cb=12&rs=1&pid=ImgDetMain&o=7&rm=3"
gold_bar_img = "https://media.istockphoto.com/id/1206809736/photo/pile-of-gold-bars.jpg?s=612x612&w=0&k=20&c=LaCbpj3_6mSKhbTd_CsUeFwqleXJFCBv9AypmbUPAvs="

with gr.Blocks(
    title="Gold Price Prediction Dashboard 💰",
    css="""
    body {
        background-color: #0e0e0e;
        font-family: 'Poppins', sans-serif;
        color: white;
    }
    .gradio-container {
        max-width: 1100px !important;
        margin: auto;
        border-radius: 20px;
        background: linear-gradient(145deg, #1b1b1b, #0e0e0e);
        box-shadow: 0 0 25px rgba(255,215,0,0.3);
        padding: 20px;
    }
    h1, h2 {
        text-align: center;
        color: gold;
        text-shadow: 0 0 10px #ffd700;
    }
    .panel {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(255,215,0,0.2);
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
    }
    """
) as demo:

    gr.HTML("<h1>🏆 Gold Price Prediction Dashboard 🥇</h1>")

    with gr.Row():
        # Left info + image panel
        with gr.Column(scale=1):
            gr.HTML(f"""
            <div class="panel">
                <img src="{gold_side_img}" style="width:100%; border-radius:15px; margin-bottom:10px;">
                <h3>📋 Prediction Inputs</h3>
                <p style="font-size:14px; color:#ddd;">
                The model predicts future gold closing prices based on current USD/INR values using multiple regression-based models.<br><br>
                Adjust the USD/INR value using the slider below and select your desired model type.
                </p>
            </div>
            """)

        # Right prediction area
        with gr.Column(scale=2):
            with gr.Row():
                usd_inr_slider = gr.Slider(70, 100, value=83, step=0.1, label="💰 USD/INR Exchange Rate", interactive=True)
                model_selector = gr.Dropdown(choices=list(model_paths.keys()), value="Linear Regression", label="🧠 Model")

            with gr.Row():
                predicted_rate_output = gr.Number(label="Predicted Gold Rate (INR)", precision=2)
                error_output = gr.Textbox(label="Logs / Info", interactive=False)

            gr.HTML(f"""
            <div style="text-align:center;">
                <img src="{gold_bar_img}" width="200" style="border-radius:10px; box-shadow:0 0 15px gold; margin-top:10px;">
                <div class="gold-box">💵 Estimated Gold Price Updates Below</div>
            </div>
            """)

            plot_output = gr.Plot(label="📊 Gold Rate Trend (Predicted)")

    # Real-time updates
    usd_inr_slider.change(
        fn=predict_gold_rate,
        inputs=[usd_inr_slider, model_selector],
        outputs=[predicted_rate_output, plot_output, error_output]
    )
    model_selector.change(
        fn=predict_gold_rate,
        inputs=[usd_inr_slider, model_selector],
        outputs=[predicted_rate_output, plot_output, error_output]
    )

    gr.HTML("""
    <footer>
        Developed ❤️ by <b>Govind Narenders</b><br>
        <a href='https://github.com/itzdineshx/Gold-price-prediction' target='_blank' style='color:gold;'>GitHub</a> |
        <a href='https://www.linkedin.com/in/dinesh-x/' target='_blank' style='color:gold;'>LinkedIn</a>
    </footer>
    """)

demo.launch(share=True)
