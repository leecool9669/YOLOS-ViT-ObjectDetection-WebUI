# -*- coding: utf-8 -*-
"""YOLOS-small 目标检测 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：YOLOS-small 已就绪（演示模式，未加载真实权重）"


def fake_detect(image, conf_threshold):
    """模拟目标检测并返回可视化描述。"""
    if image is None:
        return None, "请上传一张图片进行目标检测。"
    c = max(0.01, min(0.99, float(conf_threshold) if isinstance(conf_threshold, (int, float)) else 0.5))
    lines = [
        "[演示] 已对图像进行目标检测（未加载真实模型）。",
        f"置信度阈值：{c:.2f}",
        "检测结果示例（占位）：",
        "  - 类别: person, 边界框: [x1, y1, x2, y2], 置信度: 0.xx",
        "  - 类别: car, 边界框: [x1, y1, x2, y2], 置信度: 0.xx",
        "\n加载真实 YOLOS-small 后，将在此显示 COCO 类别与边界框。",
    ]
    return image, "\n".join(lines)


def build_ui():
    with gr.Blocks(title="YOLOS-small 目标检测 WebUI") as demo:
        gr.Markdown("## YOLOS-small 目标检测 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示基于 Vision Transformer 的 YOLOS-small 目标检测模型的典型使用流程，"
            "包括模型加载状态与图像检测结果展示。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("图像目标检测"):
                gr.Markdown("上传图片，模型将输出 COCO 类别与边界框（当前为演示模式）。")
                img_in = gr.Image(label="输入图像", type="pil")
                conf_slider = gr.Slider(
                    minimum=0.01, maximum=0.99, value=0.5, step=0.01, label="置信度阈值"
                )
                img_out = gr.Image(label="检测结果图（演示）", interactive=False)
                text_out = gr.Textbox(label="检测结果说明", lines=8, interactive=False)
                run_btn = gr.Button("执行检测（演示）")
                run_btn.click(
                    fn=fake_detect,
                    inputs=[img_in, conf_slider],
                    outputs=[img_out, text_out],
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 YOLOS-small 模型参数。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
