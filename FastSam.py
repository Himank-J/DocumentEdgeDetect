# Gradio code for running fastsam

from ultralytics import YOLO
import gradio as gr
import torch
from utils.tools_gradio import fast_process
from utils.tools import format_results, box_prompt, point_prompt, text_prompt
from PIL import ImageDraw,Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

import warnings
warnings.filterwarnings(action='ignore')

model = YOLO('./weights/FastSAM-x.pt')

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def segment_everything(
    input,
    input_size=1024, 
    withContours=True,
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    use_retina=True,
    wider=False,
    mask_random_color=True,
):
    input_size = int(input_size)  
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)

    
    annotations = results[0].masks.data
    segmented_img = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)

    bboxes = results[0].boxes.data
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    _, largest_indices = torch.topk(areas, 2)
    largest_boxes = bboxes[largest_indices]
    for i, box in enumerate(largest_boxes):
        print(f"Largest Box {i+1}: {box.tolist()}")
    print('-----------')
    
    fig, ax = plt.subplots(1)
    ax.imshow(input)
    for box in largest_boxes:
        x1, y1, x2, y2 = box[:4] 
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    cropped_img = Image.open(buf).convert("RGBA")
    cropped_img = cropped_img.resize((1024, 682))
    
    return segmented_img, cropped_img

title = "<center><strong><font size='8'>🏃 Fast Segment Anything 🤗</font></strong></center>"
description = """ # 🎯 Document edge detection using FastSam (without custom training) """
examples = [["examples/invoice3.jpeg"], ["examples/invoice2.jpeg"], ["examples/invoice1.jpeg"]]
default_example = examples[0]

input_size_slider = gr.components.Slider(minimum=512,maximum=1024,value=1024,step=64,label='Input_size',info='Our model was trained on a size of 1024')

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

demo = gr.Interface(
    segment_everything,
    inputs = [
        gr.Image(label="Input", type='pil'),
        gr.components.Slider(minimum=512,maximum=1024,value=1024,step=64,label='Input_size',info='Our model was trained on a size of 1024'),
        gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')
    ],
    outputs = [
        gr.Image(label="Segmented Image", interactive=False, type='pil'),
        gr.Image(label="Cropped Image", interactive=False, type='pil')
    ],
    title = title,
    description = description,
    examples = examples,
)
demo.launch()
