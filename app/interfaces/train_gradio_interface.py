import gradio as gr
import requests

def train_model(training_mode: str):
    """
    Function to call the train endpoint and format the results for Gradio
    """
    try:
        perform_cv = training_mode == "With Cross Validation"

        response = requests.post(
            "http://127.0.0.1:8000/train",
            json={"perform_cv": perform_cv}
        )
        if response.status_code == 200:
            data = response.json()
            return f"""
            Training Results:
            ----------------
            Status: {data['message']}
            Model saved at: {data['model_path']}
            """
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

train_interface = gr.Interface(
    fn=train_model,
    inputs=gr.Radio(
        choices=["No Cross Validation", "With Cross Validation"],
        value="With Cross Validation",
        label="Training Mode"
    ),
    outputs="text",
    title="Self Harm Predictor - Model Training",
    description="Train a new model with or without cross-validation.",
    theme="default"
)