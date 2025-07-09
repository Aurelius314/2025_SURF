import gradio as gr

with gr.Blocks() as demo:
    # 顶部标题和功能描述
    gr.Markdown("## 🖼️ 文搜图系统（原型界面）\n输入描述信息，系统展示图像。右侧为独立问答框。")

    with gr.Row():
        # 左侧内容区（搜索+图片展示）
        with gr.Column(scale=4):
            query_input = gr.Textbox(label="输入搜索内容", placeholder="例如：飞天与莲花")
            search_button = gr.Button("搜索", variant="primary")

            # 将图像按行分组，一行三个，总共21张（7行）
            for i in range(0, 20, 2):  # 每次3个一组
                with gr.Row():
                    for j in range(2):
                        if i + j < 20:
                            with gr.Column():
                                gr.Image(height=200, label=f"图片 {i + j + 1}")
                                gr.Textbox(label="Rank", value="", interactive=False)
                                gr.Textbox(label="Original Text", value="", interactive=False)
                                gr.Textbox(label="NLD Text", value="", interactive=False)

        # 右侧聊天框
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 问答助手（原型展示）")
            gr.Chatbot(label="对话内容展示区")
            gr.Textbox(label="输入对话内容", placeholder="请输入你的问题")
            gr.Button("发送")

# 启动原型界面
demo.launch(share=True)