from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("model.h5")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    # Decode base64 image
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))

    return jsonify({"prediction": digit})

# Run app (cloud compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
