import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Dummy input
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

outputs = session.run(None, {"input": input_data})

print("Prediction:", np.argmax(outputs[0]))