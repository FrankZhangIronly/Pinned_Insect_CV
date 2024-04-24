from ultralytics import YOLO
model = YOLO('yolov8n.yaml')
results = model.train(data='models/datasets/ALICE_yolov8/data.yaml',
                      device=0, epochs=100, imgsz=640, workers=1, batch=1)