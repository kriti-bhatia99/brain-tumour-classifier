# Importing the Libraries
from flask import Flask, request, render_template, Markup
import tensorflow as tf
import matplotlib.image as plt_img
import numpy as np
import os
from PIL import Image


# Global variables
app = Flask(__name__, static_folder='temp')
model = tf.keras.models.load_model('TUMOR_FINAL_MODEL.h5')
prediction_decoded = {0 : 'No Tumour', 1: 'Glioma Tumour', 2: 'Meningioma Tumour', 3: 'Pituitary Tumour'}


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Uploading the Image
        target = os.path.join("temp/")
        file = request.files['img']
        filename = file.filename
        file.save("".join([target, filename]))  

        # Resizing the Image
        filename = "temp/" + filename
        image = Image.open(filename)
        resizedImage = image.resize((128, 128))

        if resizedImage.mode != "RGB":
            resizedImage = resizedImage.convert("RGB")

        resizedImage.save(filename)

        # Reading the resized image
        data = plt_img.imread(filename)
        data = data.reshape(1, 128, 128, -1)

        # Getting the prediction
        prediction = model.predict(data)
        final_probability = np.max(prediction)
        final_probability = str(round(final_probability * 100, 2)) + "%"

        final_prediction = np.argmax(prediction, axis=1)[0]
        final_prediction = prediction_decoded[final_prediction]

        return render_template('predict.html', prediction=final_prediction + " " + str(final_probability), image=filename)

    # No file uploaded
    except IsADirectoryError:
        return render_template('index.html', error_message=Markup("<h3 class='alert alert-danger'>No File Selected</h3>"))


# Main function
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)