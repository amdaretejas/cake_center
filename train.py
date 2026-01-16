from ultralytics import YOLO
import os

# Load your already trained OBB model
model = YOLO("runs/obb/train10/weights/best.pt")

model.train(
    data=os.path.join("data", "data.yaml"),

    # ---- Training length ----
    epochs=120,
    patience=40,

    # ---- Resolution & scale handling ----
    imgsz=640,               # ↑ helps large objects
    multi_scale=False,        # VERY IMPORTANT
    batch=1,

    # ---- Learning rate (gentle fine-tuning) ----
    lr0=0.00025,             # lower than before
    lrf=0.01,
    optimizer="AdamW",

    # ---- Augmentations (BIG OBJECT SAFE) ----
    degrees=20.0,            # reduce rotation distortion
    translate=0.2,
    scale=0.8,
    shear=0.2,
    perspective=0.001,
    flipud=0.5,
    fliplr=0.5,

    # ---- OBB loss rebalance ----
    box=10,                 # reduce over-penalty
    pose=20,
    cls=0.6,
    dfl=1.2,

    # ---- Backbone adaptation ----
    freeze=4,                # unfreeze more layers

    # ---- Stability ----
    workers=0,
    amp=True,
    verbose=True,
    # resume=True
)
