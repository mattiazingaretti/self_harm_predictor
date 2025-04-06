import gradio as gr
import requests

def analyze_text(text: str, model_name: str):
    """
    Function to call the test endpoint and format the results for Gradio
    """
    try:
        response = requests.get(
            "http://127.0.0.1:8000/test",
            params={"text": text, "model_name": model_name}
        )
        if response.status_code == 200:
            data = response.json()
            return f"""
            Analysis Results:
            ----------------
            Prediction: {data['prediction_label']}
            Status: {data['message']}
            """
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

test_interface = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(
            label="Text to Analyze",
            placeholder="Enter text to analyze for potential self-harm content",
            lines=3
        ),
        gr.Textbox(
            label="Model Name",
            placeholder="Enter model name (e.g. best_rf_model.pkl)",
            value="best_rf_model.pkl"
        )
    ],
    outputs="text",
    title="Self Harm Content Analyzer",
    description="Enter text to analyze for potential self-harm content.",
    theme="default"
)