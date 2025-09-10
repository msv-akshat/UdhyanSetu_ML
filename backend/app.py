from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "saved_model"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy-load model
model = None
def get_model():
    global model
    if model is None:
        from keras.layers import TFSMLayer
        model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
    return model

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
    model_instance = get_model()
    outputs = model_instance(img_array)
    predictions = list(outputs.values())[0].numpy()
    return predictions

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
    # Local development only
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
