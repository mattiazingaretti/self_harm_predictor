import base64
from io import BytesIO
from PIL import Image
from fastapi import APIRouter
import requests
import gradio as gr

def evaluate_model(model_name: str):
    """
    Function to call the evaluate endpoint and format the results for Gradio
    """
    try:
        response = requests.post(
            "http://127.0.0.1:8000/evaluate",
            json={"model_name": model_name},
            timeout=(5,30)
        )
        if response.status_code == 200:
            data = response.json()
            
            # Format text results
            text_output = f"""
            <b>Message:</b> {data['message']}<br/>
            <b>Accuracy:</b> {data['accuracy']:.2f}<br/><br/>
            
            <b>Classification Report:</b><br/>
            <pre>{data['report']}</pre>
            """
            
            cm_image = Image.open(BytesIO(base64.b64decode(data['confusion_matrix'])))
            corr_image = Image.open(BytesIO(base64.b64decode(data['correlation_matrix'])))
            
            return (text_output, cm_image, corr_image)
            
        else:
            return (f"Error: {response.text}", None, None)
            
    except Exception as e:
        return (f"Error connecting to API: {str(e)}", None, None)


evaluate_model_interface = gr.Interface(
    fn=evaluate_model,
    inputs=gr.Textbox(
        label="Model Name",
        placeholder="Enter model name (e.g. best_rf_model_[timestamp].pkl)"
    ),
    outputs=[
        gr.HTML(label="Evaluation Results"),
        gr.Image(label="Confusion Matrix", type="pil"),
        gr.Image(label="Feature Correlation Matrix", type="pil")
    ],
    title="Self Harm Predictor - Model Evaluation",
    description="Evaluate model performance on the test dataset",
    theme="default"
)

