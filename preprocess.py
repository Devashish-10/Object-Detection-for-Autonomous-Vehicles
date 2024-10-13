import os
import cv2
import numpy as np

# Directory paths
input_dir = 'Images/images'
output_dir = 'preprocessed_images'

# New Folder for images
os.makedirs(output_dir, exist_ok=True)

# Image new dimensions
img_height, img_width = 416, 416

# Function to preprocess and save images
def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0  
    cv2.imwrite(output_path, (image * 255).astype(np.uint8))

# Preprocessing images to have images of the same dimension
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    output_path = os.path.join(output_dir, img_name)
    preprocess_image(img_path, output_path)

print("Images Preprocessed!")