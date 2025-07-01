'''
这是 Gradio 展示的用户界面文件，用于支持图像到文本和文本到图像的多模态检索功能。
python demo.py
'''
import gradio as gr
from PIL import Image
import requests
import base64
import io
from functools import partial


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
            results = result["top_k_results"]
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "rank": f"Rank {r['rank']} ({r['score']}%)", 
                    "original_text": r['original_text'],
                    "NLD_text": r['NLD_text'],
                    "image_base64": r.get('image_base64', "")
                })
            # 如果不足20条补齐空字符串
            while len(formatted_results) < 20:
                formatted_results.append({
                    "rank": "",
                    "score": "",
                    "original_text": "",
                    "NLD_text": "",
                    "image_base64": ""
                })

            return formatted_results
        else:
            return [{"original_text": f"Error: {str(e)}", "NLD_text": "", "image_base64": ""}] * 20

    except Exception as e:
        return [{
            "original_text": f"Error: {str(e)}",
            "NLD_text": "",
            "image_base64": ""
        }] * 20

    

# ---------- 2. 文搜图功能 -----------------
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
                ori = f"{r['original_text']}"
                nld = f"{r['NLD_text']}"
                rank_label = f"Rank {r['rank']} ({r['score']}%)"
                updates.extend([img, ori, nld, rank_label])
        # 不足补空
        while len(updates) < 20 * 4:
            updates.extend([None, "", "", ""])
        return updates
    
    except Exception as e:  
        print(e)  
        return [None, "", ""] * 20

# ------------图搜文 Gradio UI-------------------

#         # # 20 个文本框显示 original text 和 NLD text
#         # result_boxes = [gr.Textbox(label=f"Result {i+1}", lines=3, interactive=False) for i in range(20)]

#         # # 按钮点击后调用函数，更新20个输出框
#         # search_button.click(fn=search_image_to_text, inputs=image_input, outputs=result_boxes)

def image_to_text_tab():
    with gr.Column(elem_id="image2text-tab"):
        image_input = gr.Image(type="pil", label="Upload an Image", show_label=False)
        search_button = gr.Button("Search", variant="primary")

        output_cards = []
        for i in range(20):
            with gr.Column():
                rank_score = gr.Textbox(label="Rank", interactive=False)  # 加一个显示 Rank 和 Score 的框
                text_ori = gr.Textbox(label="Original", interactive=False)
                text_nld = gr.Textbox(label="NLD (Natural Language Description)", interactive=False)
                image_base64_box = gr.Textbox(visible=False)  # 存图像 base64
                show_button = gr.Button("Show Image")
                img = gr.Image(visible=False)
                output_cards.append((rank_score, text_ori, text_nld, image_base64_box, show_button, img))

        def handle_search(image):
            results = search_image_to_text(image)
            outputs = []
            for r in results:
                outputs.extend([
                    r.get("rank", ""),
                    r.get("original_text", ""),
                    r.get("NLD_text", ""),
                    r.get("image_base64", "")
                ])

            while len(outputs) < 20 * 4:
                outputs.extend(["", "", "", ""])
            return outputs

        flat_outputs = []
        for card in output_cards:
            flat_outputs.extend(card[:4])  # rank_score, text_ori, text_nld, image_base64_box

        search_button.click(fn=handle_search, inputs=image_input, outputs=flat_outputs)

        # ✅ 放在这里：修复 Show Image 无效的问题
        def show_image_fixed(image_base64):
            if image_base64:
                return Image.open(io.BytesIO(base64.b64decode(image_base64)))
            return None

        for card in output_cards:
            rank_score, text_ori, text_nld, image_base64_box, show_button, img = card
            show_button.click(fn=show_image_fixed, inputs=image_base64_box, outputs=img)


# ------------文搜图 Gradio UI-------------------

def text_to_image_tab():
    with gr.Column():
        query_input = gr.Textbox(label="Enter a description", placeholder="e.g., Flying apsaras with lotus flowers")
        search_button = gr.Button("Search", variant="primary")

        # 每个输出卡片是一组 [Image, original, nld]
        with gr.Row() as card_row:
            output_cards = []
            for _ in range(20): 
                with gr.Column():
                    # img = gr.Image(height=200)
                    img = gr.Image(height=200, label="Image")
                    rank_label = gr.Textbox(label="Rank", lines=1, interactive=False)
                    original = gr.Textbox(label="Original Text", lines=1, interactive=False)
                    nld = gr.Textbox(label="NLD Text (Natural Language Description)", lines=1, interactive=False)
                    output_cards.append((img, original, nld, rank_label))


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

