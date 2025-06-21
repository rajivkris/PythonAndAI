import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# Function to generate caption for the uploaded image
def generate_caption(image: np.ndarray):
    raw_image = Image.fromarray(image).convert("RGB")
    text = "A photo of a"
    inputs = processor(raw_image, text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

iface = gr.Interface(fn=generate_caption, inputs=gr.Image(type="numpy"), outputs="text", title="Image Captioning App", description="Upload an image to generate a caption for it using BLIP model.")

iface.launch(server_name="0.0.0.0", server_port=8090)