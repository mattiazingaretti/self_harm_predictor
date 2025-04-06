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
            json={"model_name": model_name}
        )
        if response.status_code == 200:
            data = response.json()
            return f"""
            Message: {data['message']}
            Accuracy: {data['accuracy']:.2f}
            
            Classification Report:
            {data['report']}
            """
        else:
            return f"Error: {response.text}"
    except Exception as e:
        print(e)
        return f"Error connecting to API: {str(e)}"


evaluate_model_interface = gr.Interface(
    fn=evaluate_model,
    inputs=gr.Textbox(
        label="Model Name",
        placeholder="Enter model name (e.g. best_rf_model_[timestamp].pkl)"
    ),
    outputs="text",
    title="Self Harm Predictor - Model Evaluation",
    description="Evaluate model performance on the test dataset",
    theme="default"
)