import gradio as gr  
from PIL import Image  
import torch  
import cn_clip.clip as clip  
from cn_clip.clip import load_from_name  
import numpy as np  


device = "cuda" if torch.cuda.is_available() else "cpu"  
model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')  
model.eval()  

def search_image_to_text(image: Image):  
    # 图像预处理  
    img_tensor = preprocess(image).unsqueeze(0).to(device)  

   
    text_candidates = ["苹果", "香蕉", "梨子", "芒果", "桃子", "葡萄", "草莓"]  

    # 编码  
    text_tensor = clip.tokenize(text_candidates).to(device)  

    with torch.no_grad():  
        # 获取图像特征  
        image_features = model.encode_image(img_tensor)  
        # 获取文本特征  
        text_features = model.encode_text(text_tensor)  
        
        # 进行归一化  
        image_features /= image_features.norm(dim=-1, keepdim=True)  
        text_features /= text_features.norm(dim=-1, keepdim=True)  

        # 计算相似度  
        logits_per_image = model.get_similarity(img_tensor, text_tensor)  
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()  

     
    percent_probs = [f"{prob * 100:.3f}%" for prob in probs[0]]  
    results = {label: prob for label, prob in zip(text_candidates, percent_probs)}  
    
    # 找到最高概率的文本  
    most_similar_text = text_candidates[np.argmax(probs)]  
    return results, most_similar_text  

def search_text_to_image(text):  
     
    return  

# UI 设计 ---------  
def image_tab():  
    with gr.Column():  
        image = gr.Image(type='pil', label="Upload an Image")  
        output_text = gr.Textbox(label="Generated Description", interactive=False)  
        button = gr.Button("Search")  
        button.click(search_image_to_text, inputs=image, outputs=[output_text])  

def text_tab():  
    with gr.Column():  
        text = gr.Textbox(label="Enter your text")  
        output_image = gr.Image(label="Generated Image", interactive=False)  
        button = gr.Button("Search")  
        button.click(search_text_to_image, inputs=text, outputs=output_image)  

with gr.Blocks() as ui:  
    gr.Markdown("<h1 style='text-align: center;'>Multi-Modal Retrieval Demo</h1>")   
    with gr.Tab("Image to Text"):  
        image_tab()  
    with gr.Tab("Text to Image"):  
        text_tab()  

ui.launch(debug=True, share=False)