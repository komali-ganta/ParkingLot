import torch
import cv2
import pandas as pd

# Load YOLOv5 model on CPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape().cpu()
model.conf = 0.25

# Load image
image_path = 'yolov5/img.jpg'  # Adjust path as needed
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
results = model(image_rgb)

# Extract results
df = results.pandas().xyxy[0]

# Filter for vehicle classes
vehicle_classes = ['car', 'truck', 'bus']
vehicles = df[df['name'].isin(vehicle_classes)]

# Save results to CSV
vehicles[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].to_csv('detected_vehicles.csv', index=False)
print("✅ CSV saved as 'detected_vehicles.csv'")

# Annotate and save image
results.save(save_dir='output_results')
print("✅ Annotated image saved in 'output_results' folder.")
