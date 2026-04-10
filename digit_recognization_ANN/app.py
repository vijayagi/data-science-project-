import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle

with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Digit Recognizer")

st.write("draw a digit in the box below or")
img=st.file_uploader("upload an image",type=["png", "jpg", "jpeg"])

canvas = st_canvas(
    fill_color='black',
    stroke_width=20,
    stroke_color='white',
    background_color='black',
    height=280,
    width=280,
    key='canvas'
)

def preprocessing(img):
    img=img.convert('L')  #converting to grayscale
    img=img.resize((28,28))
    img_array=np.array(img)
    img_array=img_array/255
    img_array=img_array.reshape(1,784)
    return img_array
if st.button('predict'):
    if img is not None:
        img = Image.open(img)
        img=preprocessing(img)
        prediction=model.predict(img)
        st.write(f"Predicted digit: {np.argmax(prediction[0])}")
        st.write(f"confidence:{np.max(prediction)}")
    elif canvas.image_data is not None:
        img = Image.fromarray((canvas.image_data).astype('uint8'))
        img=preprocessing(img)
        prediction=model.predict(img)
        st.write(f"Predicted digit: {np.argmax(prediction[0])}")
        st.write(f"confidence:{np.max(prediction)}")
    else:
        st.write("Please draw a digit!")