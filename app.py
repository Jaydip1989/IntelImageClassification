import numpy as np
from flask import Flask, render_template, redirect, request, url_for
import os
import tensorflow
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)


model = keras.models.load_model('IntelImageClassifier.h5')


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def uploads():
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    if request.method == "POST":
        f = request.files['imagefile']
        basepath = os.path.dirname(__file__)
        filename = f.filename
        image_path = os.path.join(
            basepath, '/Users/dipit/Intel Image Classification/static/uploads', filename
        )
        f.save(image_path)

        IMG = load_img(image_path, target_size=(224, 224))
        IMG = img_to_array(IMG)
        img = IMG/255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        label = np.argmax(predictions, axis=1)
        predictions_c = classes[label[0]]

        predicted_class = predictions_c.lower()
        return render_template('index.html', filename=filename, prediction=predicted_class)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)

