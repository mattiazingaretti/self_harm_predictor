from fastapi import FastAPI
from app.interfaces.evaluate_gradio_interface import evaluate_model_interface
from app.interfaces.test_gradio_interface import test_interface
from app.controllers.evaluate_controller import evaluateRouter
from app.controllers.test_controller import testRouter 
from app.controllers.train_controller import trainRouter 
from app.interfaces.train_gradio_interface import train_interface
import gradio as gr
import uvicorn

app = FastAPI(title="My FastAPI App", version="1.0.0")


with gr.Blocks(title="Self Harm Predictor") as combined_interface:
    gr.Markdown("# Self Harm Predictor")
    
    with gr.Tab("Train Model"):
        train_interface.render()
    with gr.Tab("Analyze Text"):
        test_interface.render()
    
    with gr.Tab("Evaluate Model"):
        evaluate_model_interface.render()


app = gr.mount_gradio_app(app, combined_interface, path="/gradio")
app.include_router(evaluateRouter)
app.include_router(testRouter)
app.include_router(trainRouter)

@app.get("/")
def root():
    return {
        "message": "Welcome to the Model Evaluation API",
        "endpoints": {
            "/docs": "Swagger documentation",
            "/train": "Model training endpoint",
            "/evaluate": "Model evaluation endpoint",
            "/test": "Text analysis endpoint",
            "/gradio": "Gradio interface"
        }
    }

def launch():
    uvicorn.run(app, host="127.0.0.1", port=8000)