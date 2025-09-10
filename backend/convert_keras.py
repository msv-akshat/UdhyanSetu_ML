import tensorflow as tf

# Load the SavedModel
saved_model_dir = "saved_model"
model = tf.keras.models.load_model(saved_model_dir)

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional optimizations (recommended for low memory)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_file = "model.tflite"
with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as", tflite_model_file)
