
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_model.keras"
    )


model = load_model()

st.title("Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9, then click Predict Digit.")

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=14,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    return_image_data=True,
)


    
def preprocess_digit(rgba):
    # Convert canvas to grayscale.
    # Canvas already has white strokes on a black background.
    image = Image.fromarray(rgba.astype(np.uint8)).convert("L")

    # Find the white digit without inverting the image.
    binary = image.point(lambda p: 255 if p > 30 else 0)
    bbox = binary.getbbox()

    if bbox is None:
        return None

    # Crop around the digit.
    image = image.crop(bbox)

    # Preserve proportions and fit within 20 x 20 pixels.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # MNIST-style black background.
    canvas_28 = Image.new("L", (28, 28), 0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas_28.paste(image, (x, y))

    # Center the digit by its intensity-weighted center of mass.
    arr = np.asarray(canvas_28, dtype=np.float32)
    total = arr.sum()

    if total > 0:
        yy, xx = np.indices(arr.shape)
        cx = (xx * arr).sum() / total
        cy = (yy * arr).sum() / total

        dx = int(round(13.5 - cx))
        dy = int(round(13.5 - cy))

        shifted = Image.new("L", (28, 28), 0)
        shifted.paste(image, (x + dx, y + dy))
        canvas_28 = shifted

    # Match the usual MNIST pixel scale.
    array = np.asarray(canvas_28, dtype=np.float32) / 255.0

    return array.reshape(1, 28, 28, 1), canvas_28



if st.button("Predict Digit"):
    if canvas.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        processed = preprocess_digit(canvas.image_data)

        if processed is None:
            st.warning("Please draw a digit first.")
        else:
            image_array, processed_image = processed

            # Display the exact image passed to the model.
            st.subheader("Preprocessed image")
            st.image(
                processed_image,
                width=196,
                clamp=True
            )

            probabilities = model.predict(
                image_array, verbose=0
            )[0]

            digit = int(np.argmax(probabilities))
            confidence = float(probabilities[digit]) * 100

            st.success(f"Predicted digit: {digit}")
            st.write(
                f"Model confidence score: {confidence:.2f}%"
            )

            st.subheader("Scores for digits 0–9")
            st.bar_chart({
                str(i): float(probabilities[i])
                for i in range(10)
            })

            st.caption(
                "The confidence score is the model's output score, "
                "not a guarantee that the prediction is correct."
            )
