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

'''
问答组代码
'''
API_CHAT_URL = "http://127.0.0.1:8000/chat/"  # 新增聊天API示例


# -------------- 1. 图搜文功能 ---------------
def search_image_to_text(image: Image.Image):
    # 检查图片是否为空
    if image is None:
        return [{"original_text": "Error: No image provided", "NLD_text": "", "image_base64": ""}] * 20

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

                # print("image_base64:", r.get("image_base64", ""))
                
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
            return [{"original_text": "Error: No results", "NLD_text": "", "image_base64": ""}] * 20

    except Exception as e:  
        return [{
            "original_text": f"Error: {str(e)}",
            "NLD_text": "",
            "image_base64": ""
        }] * 20

    

# ---------- 2. 文搜图功能 -----------------
def search_text_to_image(text: str):  
    if not text.strip():
        # 如果输入为空，返回20组空结果
        return [None, "", "", ""] * 20

    payload = {"query_text": text, "api_key": API_KEY}  
    try:  
        response = requests.post(API_TO_IMAGE_URL, json=payload)  
        response.raise_for_status()  
        result = response.json()  

        # 准备存储20组结果
        results = []
        if "top_k_results" in result:
            # 只取前20个结果
            for r in result["top_k_results"][:20]:  
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(r['image_base64'])))  
                    ori = f"{r['original_text']}"
                    nld = f"{r['NLD_text']}"
                    rank_label = f"Rank {r['rank']} ({r['score']}%)"
                    results.extend([img, ori, nld, rank_label])
                except Exception as e:
                    print(f"Error processing result: {e}")
                    results.extend([None, "", "", ""])
        
        # 如果结果不足20组，补充空结果
        while len(results) < 80:  # 20组 × 4个元素
            results.extend([None, "", "", ""])
            
        return results[:80]  # 确保返回正好20组结果
    
    except Exception as e:  
        print(f"Error in search_text_to_image: {e}")  
        return [None, "", "", ""] * 20

# ---------- 3. 聊天功能示例（请根据实际功能修改） -----------------
def chat_response(message, history):
    if not message.strip():
        return "", history
    
    # 构建历史对话记录
    conversation_history = []
    for user_msg, bot_msg in history:
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": bot_msg})
    
    # 添加当前用户消息
    conversation_history.append({"role": "user", "content": message})
    
    # 调用聊天API
    try:
        payload = {
            "messages": conversation_history,
            "api_key": API_KEY
        }
        response = requests.post(API_CHAT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "response" in result:
            return result["response"], history + [[message, result["response"]]]
        else:
            return "抱歉，系统无法生成回复。", history + [[message, "抱歉，系统无法生成回复。"]]

    except Exception as e:
        print(f"Error in chat_response: {e}")
        return f"系统错误: {str(e)}", history + [[message, f"系统错误: {str(e)}"]]
'''
问答组代码
'''

# ------------图搜文 Gradio UI-------------------

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
                # 添加一个隐藏的状态变量来跟踪图片是否可见
                is_visible = gr.State(False)
                toggle_button = gr.Button("Show Image")
                # 设置固定高度和宽度，保持图片美观
                img = gr.Image(visible=False, height=300, width=400)
                output_cards.append((rank_score, text_ori, text_nld, image_base64_box, is_visible, toggle_button, img))

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

        # 修改为切换图片显示状态的函数
        def toggle_image_visibility(image_base64, is_visible):
            if not is_visible:  # 如果图片当前不可见，则显示
                try:
                    if image_base64:
                        return True, Image.open(io.BytesIO(base64.b64decode(image_base64))), gr.update(visible=True), gr.update(value="Hide Image")
                except Exception as e:
                    print("解码图片失败:", e)
                return True, None, gr.update(visible=True), gr.update(value="Hide Image")
            else:  # 如果图片当前可见，则隐藏
                return False, gr.update(value=None), gr.update(visible=False), gr.update(value="Show Image")

        for card in output_cards:
            rank_score, text_ori, text_nld, image_base64_box, is_visible, toggle_button, img = card
            toggle_button.click(
                fn=toggle_image_visibility, 
                inputs=[image_base64_box, is_visible], 
                outputs=[is_visible, img, img, toggle_button]
            )


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

# ------------问答聊天框 Gradio UI 示例-------------------
def chat_interface():
    with gr.Column():
        gr.Markdown("### 🤖 问答助手")
        chatbot = gr.Chatbot(height=500, label="对话内容展示区")
        msg = gr.Textbox(label="输入对话内容", placeholder="请输入你的问题")
        send_btn = gr.Button("发送", variant="primary")
        
        # 绑定发送按钮事件
        send_btn.click(fn=chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])
        # 绑定回车键发送
        msg.submit(fn=chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])
'''
问答组代码
'''

# gradio launch
with gr.Blocks() as ui:
    gr.Markdown("<h1 style='text-align: center;'>Multi-Modal Retrieval Demo</h1>")
    
    # 使用Row布局，左侧是检索功能，右侧是聊天框
    with gr.Row():
        # 左侧检索功能区域（占比较大）
        with gr.Column(scale=4):
            with gr.Tab("Text to Image"):
                text_to_image_tab()
            with gr.Tab("Image to Text"):
                image_to_text_tab()
        
        # 右侧聊天框区域（占比较小）
        with gr.Column(scale=1):
            chat_interface()

ui.launch(debug=True, share=False)

