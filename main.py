import os
import sys
import torch
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import gradio as gr
from pyngrok import ngrok   # ✅ NEW: use pyngrok instead of Gradio’s share

# ===========================
# GPU + Versions Check
# ===========================
print("Python", sys.version.splitlines()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except:
        pass
print("Ultralytics:", ultralytics.__version__)

# ===========================
# Load model
# ===========================
trained_weights = r"runs/seg_combined/yolov8l_combined_v12/weights/best.pt"

if os.path.exists(trained_weights):
    print("✅ Loading trained weights:", trained_weights)
    model = YOLO(trained_weights)
else:
    print("⚠️ Trained weights not found. Using base pretrained yolov8l.pt")
    model = YOLO("yolov8l.pt")

# ===========================
# Dataset path
# ===========================
data_yaml = r"D:\projects\ingredient_working\data.yaml"

# ===========================
# Training
# ===========================
def train_model():
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=320,
        batch=16,
        workers=4,
        device=0,
        patience=10,
        project="runs/seg_combined",
        name="yolov8l_combined_v1"
    )

# ===========================
# Validation
# ===========================
def validate_model():
    metrics = model.val(data=data_yaml, batch=16, imgsz=640)
    print(metrics)

# ===========================
# Inference Helper
# ===========================
def predict_and_draw(model, img_path_or_array, conf=0.1, imgsz=1024):
    results = model.predict(img_path_or_array, conf=conf, imgsz=imgsz, device=0, save=False)
    r = results[0]
    im_with_overlay = r.plot()

    if hasattr(r, "boxes") and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = [model.names[i] for i in cls_ids]
    else:
        boxes, confs, names = [], [], []
    return im_with_overlay, boxes, confs, names

# ===========================
# Run Demo Inference
# ===========================
def run_demo_inference():
    img_path = "sample.jpg"  # <-- replace with your test image
    img_vis, boxes, confs, names = predict_and_draw(model, img_path, conf=0.3)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_vis)
    plt.axis('off')
    plt.show()

    print("Detections:")
    for i, (n, c, b) in enumerate(zip(names, confs, boxes)):
        x1, y1, x2, y2 = b
        print(f"{i}: {n} conf={c:.3f} bbox={[int(x1), int(y1), int(x2), int(y2)]}")

# ===========================
# Gradio App
# ===========================
def run_detection(img):
    results = model.predict(img, conf=0.25, imgsz=640, device=0)
    r = results[0]
    vis = r.plot()

    if hasattr(r, "boxes") and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        labels = [model.names[i] for i in cls_ids]
    else:
        boxes, confs, labels = [], [], []

    rows = []
    for i, (lab, conf, box) in enumerate(zip(labels, confs, boxes)):
        x1, y1, x2, y2 = box
        rows.append([i, lab, float(conf), int(x1), int(y1), int(x2), int(y2)])
    df = pd.DataFrame(rows, columns=["id", "label", "confidence", "x1", "y1", "x2", "y2"])
    return vis, df

def confirm_edits(df):
    if isinstance(df, list):
        df = pd.DataFrame(df, columns=["id","label","confidence","x1","y1","x2","y2"])
    records = df.to_dict(orient='records')
    return records

def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## Ingredient detection — upload an image and edit detections")
        with gr.Row():
            inp = gr.Image(type="numpy", label="Upload ingredient photo")
            out_img = gr.Image(label="Detections")
        out_table = gr.Dataframe(
            headers=["id","label","confidence","x1","y1","x2","y2"],
            interactive=True, 
            label="Edit labels/confidences"
        )
        btn = gr.Button("Run detection")
        confirm = gr.Button("Confirm & export list")
        btn.click(run_detection, inputs=inp, outputs=[out_img, out_table])
        confirm.click(confirm_edits, inputs=out_table, outputs=gr.JSON(label="Final confirmed detections"))

    # ✅ FIX: Use ngrok instead of share=True
    public_url = ngrok.connect(7860)
    print("🌍 Public URL:", public_url)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# ===========================
# Main Entry
# ===========================
if __name__ == "__main__":
    # train_model()
    # validate_model()
     run_demo_inference()
    #launch_gradio()
