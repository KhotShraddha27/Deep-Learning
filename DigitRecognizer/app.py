import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.title("🧠 Digit Recognizer (ONNX)")

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Upload image
uploaded_file = st.file_uploader("Upload a digit image")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")  # grayscale
        st.image(image, caption="Uploaded Image", width=150)
    except:
        st.error("Invalid image file ❌")
        st.stop()

    # 🔥 Preprocess
    image = image.resize((28, 28))
    img_array = np.array(image)

# Normalize
    img_array = img_array / 255.0

# Invert colors
    img_array = 1 - img_array

# Remove noise
    img_array = (img_array > 0.5).astype(np.float32)

# Reshape
    img_array = img_array.reshape(1, 1, 28, 28)

    # Prediction
    outputs = session.run(None, {"input": img_array})
    prediction = np.argmax(outputs[0])

    # 🔥 Convert logits → probabilities
    logits = outputs[0][0]
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores)

    probs_percent = probs * 100

    # ✅ Display results
    st.success(f"Predicted Digit: {prediction}")

    # Top 3 predictions
    top_indices = np.argsort(probs)[-3:][::-1]

    st.subheader("Top Predictions")
    for i in top_indices:
        st.write(f"Digit {i}: {probs_percent[i]:.2f}%")

    # Confidence chart
    st.subheader("Confidence Scores (%)")
    st.bar_chart(probs_percent)

    confidence = np.max(probs_percent)
    st.info(f"Model Confidence: {confidence:.2f}%")