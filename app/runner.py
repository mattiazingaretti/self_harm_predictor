from fastapi import FastAPI
from app.interfaces.evaluate_gradio_interface import evaluate_model
from app.controllers.evaluate_controller import router
import gradio as gr
import uvicorn

app = FastAPI(title="My FastAPI App", version="1.0.0")

gradio_app = gr.Interface(
    fn=evaluate_model,
    inputs=gr.Textbox(
        label="Model Name",
        placeholder="Enter model name (e.g. best_rf_model)",
        value="best_rf_model"
    ),
    outputs="text",
    title="Self Harm Predictor - Model Evaluation",
    description="Evaluate model performance on the test dataset",
    theme="default"
)

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Model Evaluation API",
        "endpoints": {
            "/docs": "Swagger documentation",
            "/evaluate": "Model evaluation endpoint",
            "/gradio": "Gradio interface"
        }
    }

def launch():
    uvicorn.run(app, host="127.0.0.1", port=8000)