import gradio as gr
from bs4 import BeautifulSoup as bs
from io import BytesIO
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption_from_url(image_url: str):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        text = "Image shows a"
        inputs = processor(image, text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    
def extract_image_from_url(url: str):
    try:
        response = requests.get(url)
        soup = bs(response.content, "html.parser")
        img_tags = soup.find_all("img")
        with open("captions.txt", 'w') as file:
            for img in img_tags:
                img_url = img.get("src")
                if img_url and img_url.startswith("//"):
                    img_url = "https:" + img_url
                if img_url and 'svg' not in img_url and '1x1' not in img_url and (img_url.startswith("http") or img_url.startswith("https")):
                    caption = generate_caption_from_url(img_url)
                    file.write(f"Image URL: {img_url}\n Caption: {caption}\n-------------------------------------------------\n")
                    return f"Captions are save to {file.name}"
    except Exception as e:
        print(f"Error extracting images: {str(e)}")

print(extract_image_from_url("https://en.wikipedia.org/wiki/IBM"))