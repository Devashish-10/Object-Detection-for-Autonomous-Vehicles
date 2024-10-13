import os
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from ultralytics import YOLO

# Configuration
input_dir = 'Datasets'
labels_file = os.path.join(input_dir, 'Labels/labels_train.csv')
images_dir = 'preprocessed_images/'
output_labels_path = os.path.join(input_dir, 'labels_yolo')
num_train_images = 1000
epochs = 20

def load_dataset(labels_file, images_dir, num_images):
    """Load a subset of the dataset for training."""
    df = pd.read_csv(labels_file)
    df = shuffle(df)
    selected_indices = np.random.choice(len(df), size=num_images, replace=False)
    selected_df = df.iloc[selected_indices]

    train_images = []
    train_boxes = []

    for _, row in selected_df.iterrows():
        image_path = os.path.join(images_dir, row['frame'])
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        train_images.append(image)
        train_boxes.append([row['xmin'], row['xmax'], row['ymin'], row['ymax']])

    if len(train_images) == 0 or len(train_boxes) == 0:
        raise ValueError("No images or bounding boxes were loaded. Check the dataset and file paths.")

    return train_images, train_boxes

def train_yolo_model():
    print("Loading dataset...")
    train_images, train_boxes = load_dataset(labels_file, images_dir, num_train_images)

    print("Initializing YOLO model...")
    model = YOLO("yolov5s.pt")

    print(f"Training on {num_train_images} images for {epochs} epochs...")

    # Ensure output_labels_path directory exists
    os.makedirs(output_labels_path, exist_ok=True)

    # Train the model using 'train' mode and data.yaml configuration
    model.train(task='train', data='data.yaml', epochs=epochs)

    print("Training complete!")

    # Save the trained model
    save_path = "trained_yolo_model.pt"
    model.save(save_path)
    print(f"Trained model saved at: {save_path}")

if __name__ == "__main__":
    train_yolo_model()
