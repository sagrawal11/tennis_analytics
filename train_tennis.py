from ultralytics import YOLO

# Load pre-trained YOLOv8 medium model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(
    data='data.yaml',        # path to your dataset config
    epochs=50,              # number of training epochs
    imgsz=640,               # image size
    batch=6,                 # batch size (adjust based on M1 memory)
    device='mps',            # use M1 GPU acceleration
    name='tennis_detection', # experiment name
    save=True,               # save training checkpoints
    plots=True,              # generate training plots
    patience=10,             # early stopping patience
    amp=True,                # automatic mixed precision
    mixup=0.1,              # mixup augmentation
    copy_paste=0.1          # copy-paste augmentation
)

# Evaluate the model
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
