from flask import Flask, render_template, request, redirect, url_for,flash
import os
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Path to the trained RNN model
model_path = 'trained_models/trained_rnn_model.h5'

# Load the trained RNN model
model = load_model(model_path)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded image with the model
            result_image_path = process_image(filepath)
            
            # Display the result on a new page or modal
            return render_template('result.html', result_image=result_image_path, filename=filename)
    
    return render_template('index.html')

# Function to process image with the model
def process_image(image_path):
    image = cv2.imread(image_path)
    
    # Preprocess image if necessary
    # Example: Resize to match model input size
    # image = cv2.resize(image, (model_input_width, model_input_height))
    
    # Perform prediction with the model
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # Example code to visualize predictions
    for prediction in predictions:
        xmin, xmax, ymin, ymax = prediction  # Adjust according to your model output format
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    # Save the processed image with bounding boxes
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_image_path, image)
    
    return result_image_path

if __name__ == '__main__':
    app.run(debug=True)
