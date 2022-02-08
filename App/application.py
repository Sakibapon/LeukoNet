from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, app
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS,cross_origin

# Define a flask app
application = Flask(__name__)
app=application
@app.route('/', methods=['GET'])
@cross_origin()
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def upload():

    if request.method == 'POST':

        model = load_model('WhiteBloodCellModel.hdf5')

        class_labels = [
            'ALL',
            'AML',
            'CML',
            'CLL',
            'MDS',
            'Healthy',
            'Lymphoma',
            'Pre',
            'Pro',
            'Begining'
        ]
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255
        pred = np.argmax(model.predict(x))
        print(pred)
        result = class_labels[pred]
        # Make prediction
        # preds = model_predict(file_path, model)
        # result=class_labels([preds])

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=8080)
    app.run()