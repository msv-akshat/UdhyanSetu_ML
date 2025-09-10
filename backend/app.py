from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.tflite"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy-load TFLite interpreter
interpreter = None
input_details = None
output_details = None

def get_interpreter():
    global interpreter, input_details, output_details
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

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

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict_image(img_array):
    interpreter, input_details, output_details = get_interpreter()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["image"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    predictions = predict_image(img_array)
    os.remove(file_path)

    class_index = np.argmax(predictions[0])
    scores = predictions[0]
    top_indices = np.argsort(scores)[-3:][::-1]
    top_classes = [(CLASS_NAMES[i], float(scores[i])) for i in top_indices]

    res = {"prediction": top_classes[0][0], "top_3": top_classes}
    return jsonify(res)

@app.route("/")
def home():
    return "🌱 Plant disease detection API is running!"

@app.route("/healthz")
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
