from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from flask import render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads') 
# Load the trained model
model = tf.keras.models.load_model('my_model2.h5')

# Function to preprocess and predict image
def import_and_predict(image_data, model):
    image = Image.open(image_data)
    image = ImageOps.fit(image, (100, 100), Image.LANCZOS)
    image = image.convert('RGB')
    image_array = np.asarray(image)
    image_array = (image_array.astype(np.float32) / 255.0)
    img_reshape = image_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction, image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Check if the file exists in Glaucomatous or Healthy folders
            filename = file.filename
            glaucoma_dir = os.path.join('Glaucomatous', filename)
            healthy_dir = os.path.join('Healthy', filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("Image saved at:", file_path)

            if os.path.exists(glaucoma_dir):
                image_path = glaucoma_dir
            elif os.path.exists(healthy_dir):
                image_path = healthy_dir
            else:
                return '<p style="color: red; font-weight: bold;">Invalid image file selected.</p>'

            # Make prediction
            prediction, image = import_and_predict(file_path, model)
            pred = prediction[0][0]
            pred_percentage = "{:.2f}".format(pred * 100)
            if pred > 0.5:
                result = "Your eye is Healthy. Great!!"
            else:
                result = "You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible."

            # Pass the image path and prediction result to the result template
            return render_template('result.html', result=result, image_path=file_path, pred_percentage=pred_percentage, os=os)
            # return render_template('result.html', result=result)
        
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
