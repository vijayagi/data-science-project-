import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("🔢 Digit Recognizer")
st.markdown("Draw or upload a digit (0–9) and get a prediction!")

# Layout
col1, col2 = st.columns(2)

# -------- LEFT: DRAW --------
with col1:
    st.subheader("✏️ Draw Digit")
    canvas = st_canvas(
        fill_color='black',
        stroke_width=20,
        stroke_color='white',
        background_color='black',
        height=280,
        width=280,
        key='canvas'
    )

# -------- RIGHT: UPLOAD --------
with col2:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# -------- PREPROCESS FUNCTION --------
def preprocessing(img):
    img = img.convert('L')  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)

    processed_display = img_array  # for showing

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, processed_display

# -------- PREDICT BUTTON --------
if st.button("🚀 Predict"):

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        st.subheader("🖼️ Uploaded Image")
        st.image(img, width=200)

        processed, display_img = preprocessing(img)

    elif canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype('uint8'))

        st.subheader("🖼️ Drawn Image")
        st.image(img, width=200)

        processed, display_img = preprocessing(img)

    else:
        st.warning("⚠️ Please draw or upload an image first!")
        st.stop()

    # preprocessing image
    st.subheader("⚙️ Preprocessed Image (28x28)")
    st.image(display_img, width=150, clamp=True)

    # Prediction
    prediction = model.predict(processed)

    digit = np.argmax(prediction[0])
    confidence = np.max(prediction)

    # Display result
    st.markdown("## 🎯 Prediction Result")
    st.success(f"Predicted Digit: **{digit}**")
    st.info(f"Confidence: **{confidence:.2f}**")