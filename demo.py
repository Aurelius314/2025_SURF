'''
这是 Gradio 展示的用户界面文件，用于支持图像到文本和文本到图像的多模态检索功能。

'''
import gradio as gr
from PIL import Image
import requests
import base64
import io

# API Key（模拟）
API_KEY = "your_demo_api_key"
API_TO_TEXT_URL = "http://127.0.0.1:8000/image-to-text/"
API_TO_IMAGE_URL = "http://127.0.0.1:8000/text-to-image/"


# -------------- 1. 图搜文功能 ---------------

def search_image_to_text(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    payload = {"img_base64": img_base64, "api_key": API_KEY}
    try:
        response = requests.post(API_TO_TEXT_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        if "top_k_results" in result:
            formatted = "\n".join(
                [f"rank {r['rank']}: {r['text']} - {r['score']}%" for r in result['top_k_results']])
            return formatted
        else:
            return "No matching result."
        
    except Exception as e:
        return f"Error: {str(e)}"
    

# ---------- 2. 文搜图 -----------------

def search_text_to_image(text: str):  
    payload = {"query_text": text, "api_key": API_KEY}  
    try:  
        response = requests.post(API_TO_IMAGE_URL, json=payload)  
        response.raise_for_status()  
        result = response.json()  

        if "top_k_results" in result:  
            cards = []  
            for r in result['top_k_results']:  
                img = Image.open(io.BytesIO(base64.b64decode(r['image_base64'])))  
                cards.append((img, f"rank {r['rank']}: {r['score']}% - tags: {r['tags']}"))   
            return cards  
        return []  
    except Exception as e:  
        print(e)  
        return []

# Gradio UI


def image_to_text_tab():
    with gr.Column():
        image = gr.Image(type='pil', label="Upload an Image")
        output = gr.Textbox(label="Top 5 Similar Texts", interactive=False, lines=6)
        button = gr.Button("Search")
        button.click(search_image_to_text, inputs=image, outputs=output)


def text_to_image_tab():
    with gr.Column():
        text_input = gr.Textbox(label="Enter a description")
        gallery = gr.Gallery(label="Top 5 Similar Images", show_label=True, columns=5)
        button = gr.Button("Search")
        button.click(search_text_to_image, inputs=text_input, outputs=gallery)


with gr.Blocks() as ui:
    gr.Markdown("<h1 style='text-align: center;'>Multi-Modal Retrieval Demo</h1>")
    with gr.Tab("Image to Text"):
        image_to_text_tab()
    with gr.Tab("Text to Image"):
        text_to_image_tab()

ui.launch(debug=True, share=False)

