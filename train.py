import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define paths
images_path = 'Datasets/preprocessed_images'

# Configure training settings for limited GPU memory
train_settings = {
    "data": "data.yaml",
    "epochs": 40,
    "imgsz": 416,
    "batch": 32,  
}

# Load the YOLO model
model = YOLO("yolov5s.pt")

# Train the model
results = model.train(**train_settings)

# Save the trained model
model.save("trained_model.pt")

# Perform inference on selected images
# Replace with your specific selection method
selected_images = ['image1.jpg', 'image2.jpg']

image_paths = [os.path.join(images_path, img) for img in selected_images]

results = model(image_paths)

# Process and display results
for result, image_path in zip(results, image_paths):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for box in result.xyxy:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()