import os
import pandas as pd
import cv2

# Define paths
images_path = 'preprocessed_images'
labels_path = 'Datasets/Labels'
output_labels_path = 'Datasets/labels_yolo'

# Create output labels directory if it doesn't exist
os.makedirs(output_labels_path, exist_ok=True)

# Load CSV data
labels_df = pd.read_csv(os.path.join(labels_path, 'labels_train.csv'))

# Define the YOLO class label mapping (adjust as needed)
class_mapping = {
    1: 'car',
    2: 'truck',
    3: 'person',
    4: 'bicycle',
    5: 'traffic light',
}

# Function to convert CSV data to YOLO format
def convert_to_yolo(df, output_dir):
    for _, row in df.iterrows():
        frame = row['frame']
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        class_id = row['class_id']

        # Map class_id to YOLO label
        if class_id not in class_mapping:
            continue
        
        yolo_class_id = list(class_mapping.keys()).index(class_id)

        # Calculate YOLO format values
        img_path = os.path.join(images_path, frame)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        # Ensure bounding box coordinates are within [0, 1]
        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
            print(f"Skipping invalid label for {frame}")
            continue

        # Create or overwrite label file
        label_filename = os.path.splitext(frame)[0] + '.txt'
        label_filepath = os.path.join(output_dir, label_filename)
        with open(label_filepath, 'w') as f:
            f.write(f"{yolo_class_id} {x_center} {y_center} {width} {height}\n")

# Convert labels to YOLO format
convert_to_yolo(labels_df, output_labels_path)

# Create data.yaml file
data_yaml = f"""
train: {os.path.abspath(images_path)}
val: {os.path.abspath(images_path)}

nc: 5  # number of classes, adjust as needed
names: ['car', 'truck', 'person', 'bicycle', 'traffic light']  # replace with your class names
"""

# Save data.yaml file
with open('data.yaml', 'w') as file:
    file.write(data_yaml)
print("Labelation Complete")