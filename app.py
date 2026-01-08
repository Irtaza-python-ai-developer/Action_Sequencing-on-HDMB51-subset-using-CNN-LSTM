import gradio as gr
from PIL import Image
from src.inference import predict_action

def run(image):
    annotated, label = predict_action(image)
    return annotated, label

iface = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.Textbox(label="Predicted Action")
    ],
    title="Action Recognition using CNN + LSTM",
    description="Upload an image. Model predicts action using CNN + LSTM."
)

iface.launch()
