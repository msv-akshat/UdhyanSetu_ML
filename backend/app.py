from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = "saved_model"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Healthy",
    "Bacterial Leaf Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Yellow Leaf Curl Virus",
    "Powdery Mildew",
    "Downy Mildew"
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["image"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess
    img_array = preprocess_image(file_path)
    
    # Predict
    predictions = model.predict(img_array)  # Already a NumPy array
    class_index = np.argmax(predictions[0])
    class_name = CLASS_NAMES[class_index]
    
    # Get top 3 predictions
    scores = predictions[0]  # directly use it
    top_indices = np.argsort(scores)[-3:][::-1]
    top_classes = [(CLASS_NAMES[i], float(scores[i])) for i in top_indices]

    # Optionally, remove uploaded file
    os.remove(file_path)

    res = {"prediction": top_classes[0][0], "top_3": top_classes}
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
