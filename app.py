"""Handwritten Digit Identifier: a Streamlit app around a CNN trained on MNIST.

Upload (or photograph) a single handwritten digit, 0-9, and the model predicts it.
The first run trains the CNN on MNIST (about 1-2 minutes) and caches it.
"""
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Identifier", page_icon="✍️", layout="centered")


# ---------------------------------------------------------------- model ----
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved model if it exists, otherwise train a CNN on MNIST."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    model.test_accuracy = float(acc)
    try:
        model.save(MODEL_PATH)
    except Exception:
        pass  # saving is optional; the cached model still works for this session
    return model


# ------------------------------------------------------- preprocessing ----
def preprocess(img: Image.Image) -> np.ndarray:
    """Turn a photo of a digit into a 28x28 white-on-black array like MNIST.

    Steps: grayscale -> auto-contrast -> invert if the paper is light ->
    crop to the digit -> fit into 20x20 -> centre on a 28x28 canvas.
    """
    gray = ImageOps.exif_transpose(img).convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    arr = np.asarray(gray, dtype=np.float32)

    if arr.mean() > 127:  # dark ink on light paper -> flip to white on black
        arr = 255.0 - arr

    # keep only strong strokes so paper shading doesn't count as ink
    arr[arr < 0.25 * arr.max()] = 0
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    digit = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = np.asarray(Image.fromarray(digit.astype("uint8")).resize((new_w, new_h), Image.LANCZOS), dtype=np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = digit

    # shift so the centre of mass sits in the middle, as in MNIST
    total = canvas.sum()
    if total > 0:
        cy, cx = (np.indices(canvas.shape) * canvas).sum(axis=(1, 2)) / total
        canvas = np.roll(canvas, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))
    return canvas / 255.0


# ------------------------------------------------------------------ UI ----
st.title("✍️ Handwritten Digit Identifier")
st.write(
    "Write **one digit (0-9)** in dark ink on plain paper, take a clear photo, and upload it. "
    "A Convolutional Neural Network trained on the 70,000-image MNIST dataset will tell you what it sees."
)

with st.spinner("Getting the model ready. The first launch trains it and takes 1-2 minutes..."):
    model = load_model()

if hasattr(model, "test_accuracy"):
    st.caption(f"Model accuracy on MNIST test images: {model.test_accuracy:.2%}")

tab_upload, tab_camera = st.tabs(["Upload a photo", "Use the camera"])
with tab_upload:
    uploaded = st.file_uploader("Choose an image of a digit", type=["jpg", "jpeg", "png"])
with tab_camera:
    snapshot = st.camera_input("Take a photo of a digit")

source = uploaded or snapshot
if source is not None:
    image = Image.open(source)
    processed = preprocess(image)
    probs = model.predict(processed[np.newaxis, ..., np.newaxis], verbose=0)[0]
    top = int(np.argmax(probs))

    left, right = st.columns(2)
    left.image(image, caption="Your image", width="stretch")
    right.image(processed, caption="What the model sees (28x28)", clamp=True, width=200)

    st.subheader(f"Prediction: {top}")
    st.write(f"Confidence: **{probs[top]:.1%}**")
    st.bar_chart({str(i): float(p) for i, p in enumerate(probs)}, y_label="Probability", x_label="Digit")
    if probs[top] < 0.6:
        st.info("The model isn't very sure. Try a clearer photo: one thick, dark digit on plain paper, with good light.")

with st.expander("How it works"):
    st.markdown(
        """
1. **Train:** a CNN (two Conv2D + MaxPooling blocks, Dropout, and a softmax Dense layer) learns from MNIST.
2. **Preprocess:** your photo is turned to grayscale, inverted to white-on-black, cropped to the digit,
   resized to fit 20x20, and centred on a 28x28 canvas, which is the format MNIST uses.
3. **Predict:** the model outputs a probability for each digit 0-9, and the highest one wins.
"""
    )
st.caption("Built by Eric Victory Onyinyechi | Python, TensorFlow/Keras, Pillow, NumPy, Streamlit")
