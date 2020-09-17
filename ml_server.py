# Removing warnings
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

#saved_model = 'model.h5'
#saved_model = 'mnist-model.h5'
#saved_model = 'model-relu.h5'

model = tf.keras.models.load_model('model-relu.h5')
feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
_, (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.

def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :] # x_test shape :  (10000, 28, 28)
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image


@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        preds, image = get_prediction()
        final_preds = [p.tolist() for p in preds]
        return json.dumps({
            'prediction': final_preds,
            'image': image.tolist()
        })
    return 'Welcome to the model server!'

if __name__ == '__main__':
    app.run()
