from ultralytics import YOLO
import os

# Load PRETRAINED OBB model (important)
model = YOLO("yolo11n-obb.pt")

model.train(
    data= os.path.join("data", "data.yaml"),  # dataset path

    # ---- Core training ----
    epochs=200,              # small data → more epochs
    imgsz=640,
    batch=4,                 # small batch for stability
    # device=0,                # GPU (use "cpu" if needed)

    # ---- Learning rate tuning (VERY IMPORTANT) ----
    lr0=0.0003,              # low LR for fine-tuning
    lrf=0.01,
    optimizer="AdamW",

    # ---- Regularization ----
    weight_decay=0.0005,
    patience=50,             # early stopping

    # ---- Augmentations (angle-focused) ----
    degrees=45.0,            # strong rotation
    translate=0.05,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,              # avoid flipping if angle matters
    fliplr=0.0,

    # ---- OBB specific ----
    box=7.5,                 # bbox loss weight ↑
    cls=0.5,
    dfl=1.5,

    # ---- Stability ----
    freeze=10,               # freeze backbone layers
    workers=2,
    amp=True,
    verbose=True
)
