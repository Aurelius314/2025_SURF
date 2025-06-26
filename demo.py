'''
这是 Gradio 展示的用户界面文件，用于支持图像到文本和文本到图像的多模态检索功能。
python demo.py
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
''' 后端输出 
{
  "top_k_results": [
    {
      "rank": 1,
      "text_id": "...",
      "original_text": "...",
      "NLD_text": "...",
      "score": ...
    },
    ...
  ]
}

'''
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
            results = result["top_k_results"]
            formatted_results = []
            for r in results:
                text = f"Rank {r['rank']} ({r['score']}%)\nOriginal: {r['original_text']}\nNLD: {r['NLD_text']}"
                formatted_results.append(text)
            # 如果不足9条补齐空字符串
            while len(formatted_results) < 9:
                formatted_results.append("")
            return formatted_results
        else:
            return ["No matching result."] * 9

    except Exception as e:
        return [f"Error: {str(e)}"] * 9
    

# ---------- 2. 文搜图功能 -----------------
'''
后端输出
{
  "rank": 1,
  "image_id": "...",
  "text_id": "...",
  "original_text": "...",
  "NLD_text": "...",
  "image_base64": "...",
  ...
}
'''
def search_text_to_image(text: str):  
    payload = {"query_text": text, "api_key": API_KEY}  
    try:  
        response = requests.post(API_TO_IMAGE_URL, json=payload)  
        response.raise_for_status()  
        result = response.json()  

        updates = []
        if "top_k_results" in result:  
            for r in result["top_k_results"]:  
                img = Image.open(io.BytesIO(base64.b64decode(r['image_base64'])))  
                ori = f"Original: {r['original_text']}"
                nld = f"NLD: {r['NLD_text']}"
                updates.extend([img, ori, nld])
        # 不足补空
        while len(updates) < 9 * 3:
            updates.extend([None, "", ""])
        return updates
    
    except Exception as e:  
        print(e)  
        return [None, "", ""] * 9

# ------------图搜文 Gradio UI-------------------

def image_to_text_tab():
    with gr.Column(elem_id="image2text-tab"):

        image_input = gr.Image(type="pil", label="Upload an Image", show_label=False)
        search_button = gr.Button("Search")

        # 9 个文本框显示 original text 和 NLD text
        result_boxes = [gr.Textbox(label=f"Result {i+1}", lines=3, interactive=False) for i in range(9)]

        # 按钮点击后调用函数，更新9个输出框
        search_button.click(fn=search_image_to_text, inputs=image_input, outputs=result_boxes)



# ------------文搜图 Gradio UI-------------------

def text_to_image_tab():
    with gr.Column():
 
        query_input = gr.Textbox(label="Enter a description", placeholder="e.g., Flying apsaras with lotus flowers")
        search_button = gr.Button("Search")

        # 每个输出卡片是一组 [Image, original, nld]
        with gr.Row() as card_row:
            output_cards = []
            for _ in range(9):  # 假设只显示前 5 个
                with gr.Column():
                    img = gr.Image(height=200)
                    original = gr.Textbox(label="Original Text", lines=1, interactive=False)
                    nld = gr.Textbox(label="NLD Text", lines=1, interactive=False)
                    output_cards.append((img, original, nld))


        # 把扁平的 output_components 列出来绑定
        flat_outputs = [comp for card in output_cards for comp in card]
        search_button.click(fn=search_text_to_image, inputs=[query_input], outputs=flat_outputs)

# gradio launch
with gr.Blocks() as ui:
    gr.Markdown("<h1 style='text-align: center;'>Multi-Modal Retrieval Demo</h1>")
    with gr.Tab("Image to Text"):
        image_to_text_tab()
    with gr.Tab("Text to Image"):
        text_to_image_tab()

ui.launch(debug=True, share=False)

