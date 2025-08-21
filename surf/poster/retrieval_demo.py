'''
这是集成多模态检索和知识问答的前端demo,需要先启动cnclip_api_service,然后启动这个文件
python retrieval_demo.py
'''
import gradio as gr
from PIL import Image
import requests
import base64
import io
import os
from functools import partial

# 检索 API Key
API_KEY = "surf_demo_api_key"
API_TO_TEXT_URL = "http://127.0.0.1:8002/image-to-text/"
API_TO_IMAGE_URL = "http://127.0.0.1:8002/text-to-image/"


# ------------------图搜文功能（返回 rank、score 与完整 record 字段）------------------
def search_image_to_text(image: Image.Image):
    if image is None:
        return [{"rank": "", "score": "", "record": {}} for _ in range(20)]

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    payload = {
        "query_image": img_base64,
        "api_key": API_KEY,
        "offset": 0,
        "limit": 20
    }
    try:
        response = requests.post(API_TO_TEXT_URL, json=payload)
        if not response.ok:
            try:
                print("image-to-text error response:", response.text[:1000])
            except Exception:
                pass
            response.raise_for_status()
        result = response.json()

        # 兼容两种返回字段：results 或 top_k_results
        if "results" in result:
            raw_results = result["results"]
        elif "top_k_results" in result:
            raw_results = result["top_k_results"]
        else:
            raw_results = []
        formatted_results = []
        for r in raw_results:
            formatted_results.append({
                "rank": r.get("rank", ""),
                "score": r.get("score", ""),
                # 兼容可能直接将 table/fields 平铺在同级或嵌套在 record
                "record": r.get("record", r)
            })

        while len(formatted_results) < 20:
            formatted_results.append({"rank": "", "score": "", "record": {}})

        return formatted_results[:20]

    except Exception as e:
        results = [{"rank": "", "score": "", "record": {}} for _ in range(20)]
        results[0] = {"rank": "", "score": "", "record": {"error": str(e)}}
        return results

    
# -----------------文搜图功能------------------
def search_text_to_image(text: str):  
    if not text.strip():
        # 返回 20 组占位：[image, rank_score]
        placeholder = []
        for i in range(20):
            placeholder.extend([None, gr.update(label=f"rank {i+1}", value="")])
        return placeholder

    payload = {"query_text": text, "api_key": API_KEY}
    try:
        response = requests.post(API_TO_IMAGE_URL, json=payload)
        if not response.ok:
            try:
                print("text-to-image error response:", response.text[:1000])
            except Exception:
                pass
            response.raise_for_status()
        result = response.json()

        outputs = []  # 扁平输出：[img, rank_score] * N
        raw_results = []
        if "results" in result:
            raw_results = result["results"]
        elif "top_k_results" in result:
            raw_results = result["top_k_results"]

        for idx, r in enumerate(raw_results[:20]):
            rank_value = r.get("rank", idx + 1)
            score_value = r.get("score", "")

            img_obj = None
            rec = r.get("record", r)
            fields = rec.get("fields", rec)
            img_path = fields.get("local_path")
            try:
                # 仅从本地路径加载图片
                if img_path and os.path.exists(img_path):
                    img_obj = Image.open(img_path)
            except Exception as ie:
                print(f"Error loading image for rank {rank_value}: {ie}")
                img_obj = None

            outputs.extend([
                img_obj,
                gr.update(label=f"rank {rank_value}", value=str(score_value))
            ])

        # 不足 20 组补齐
        for j in range(len(outputs) // 2, 20):
            outputs.extend([None, gr.update(label=f"rank {j+1}", value="")])

        return outputs[:40]

    except Exception as e:
        print(f"Error in search_text_to_image: {e}")
        outputs = [None, gr.update(label="rank 1", value=f"Error: {str(e)}")]
        for i in range(1, 20):
            outputs.extend([None, gr.update(label=f"rank {i+1}", value="")])
        return outputs

# ===图搜文 Gradio UI===
def image_to_text_tab():
    with gr.Column(elem_id="image2text-tab"):
        image_input = gr.Image(
            type="pil",
            label="Upload an Image",
            show_label=False,
            height=320,
            width=480,
            elem_id="query-image-input"
        )
        search_button = gr.Button("Search", variant="primary")

        output_cards = []
        for i in range(20):
            with gr.Column():
                rank_score_box = gr.Textbox(label="rank", lines=1, interactive=False)
                # record_box = gr.JSON(label="Record", value={}, visible=True)  # 保留旧版：展示完整 record JSON
                lang_desc_box = gr.Textbox(label="language description", lines=1, interactive=False)
                # output_cards.append((rank_score_box, record_box))  # 旧版
                output_cards.append((rank_score_box, lang_desc_box))  # 新版：展示语言描述

        def handle_search(image):
            results = search_image_to_text(image)
            outputs = []
            # 新版：仅输出 [rank+score(Textbox update), language description(Textbox update)]
            for idx, r in enumerate(results[:20]):
                rec = r.get('record', {})
                fields = rec.get('fields', rec)
                desc = fields.get('nld_text') or fields.get('language_description') or ""
                outputs.extend([
                    gr.update(label=f"rank {r.get('rank', '')}", value=f"{r.get('score', '')}"),
                    gr.update(label="language description", value=str(desc))
                ])
            # 旧版：输出 record JSON
            # for idx, r in enumerate(results[:20]):
            #     outputs.extend([
            #         gr.update(label=f"rank {r.get('rank', '')}", value=f"{r.get('score', '')}"),
            #         r.get('record', {})
            #     ])
            # 不足 20 组时补齐
            current_pairs = len(outputs) // 2
            for j in range(current_pairs, 20):
                outputs.extend([gr.update(label=f"rank {j+1}", value=""), gr.update(label="language description", value="")])
            return outputs[: 20 * 2]

        flat_outputs = []
        for card in output_cards:
            flat_outputs.extend(card)  # rank+score, language description

        search_button.click(fn=handle_search, inputs=image_input, outputs=flat_outputs)


# ===文搜图 Gradio UI===
def text_to_image_tab():
    with gr.Column():
        query_input = gr.Textbox(label="Enter a description", placeholder="e.g., 九色鹿")
        search_button = gr.Button("Search", variant="primary")

        # 每个输出卡片是一组 [Image, rank+score]，保持原行列布局
        with gr.Row() as card_row:
            output_cards = []
            for _ in range(20): 
                with gr.Column():
                    img = gr.Image(height=200, label="Image")
                    rank_score_box = gr.Textbox(label="rank", lines=1, interactive=False)
                    output_cards.append((img, rank_score_box))

        # 扁平输出组件并绑定
        flat_outputs = [comp for card in output_cards for comp in card]
        search_button.click(fn=search_text_to_image, inputs=[query_input], outputs=flat_outputs)




# *** gradio launch ***
with gr.Blocks(css="""
#image2text-tab .image-frame { max-width: 520px; }
#image2text-tab .image-frame img { max-height: 340px; width: auto; }
""") as ui:
    gr.Markdown("<h1 style='text-align: center;'>Multi-Modal Retrieval Demo</h1>")
    with gr.Tab("Image to Text"):
        image_to_text_tab()
    with gr.Tab("Text to Image"):
        text_to_image_tab()

ui.launch(debug=True, share=False)
