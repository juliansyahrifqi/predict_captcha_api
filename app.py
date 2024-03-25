from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from dotenv import load_dotenv

app = Flask(__name__)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

MAX_LENGTH = 6
IMG_WIDTH = 170
IMG_HEIGHT = 40

load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.h5")

# Load the model with compile=False
model = keras.models.load_model(
    MODEL_PATH, custom_objects={"CTCLayer": CTCLayer}, compile=False
)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

vocab_path = os.path.join(PROJECT_ROOT, "vocab.txt")
with open(vocab_path, "r") as f:
    vocab = f.read().splitlines()

num_to_char = layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True)

class CaptchaController:
    @staticmethod
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LENGTH]
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    @staticmethod
    def classify_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.expand_dims(img, axis=0)
        preds = prediction_model.predict(img)
        pred_text = CaptchaController.decode_batch_predictions(preds)
        return pred_text[0]

@app.route('/predict_captcha', methods=['POST'])
def predict_captcha():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image.save(f"{PROJECT_ROOT}/captcha.png")  # Save the image to a temporary file

    predicted_captcha = CaptchaController.classify_image(f"{PROJECT_ROOT}/captcha.png")

    return jsonify({'predicted_captcha': predicted_captcha}), 200

if __name__ == '__main__':
    app.run(port=9000, debug=True)
