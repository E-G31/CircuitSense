import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="CircuitSense",layout="centered")

st.title("CircuitSense")

st.markdown(
"""
### AI-Powered Handwritten Circuit Symbol Recognition

Upload a handwritten circuit symbol and let CircuitSense identify it instantly.
"""
)

st.sidebar.title("About")

st.sidebar.info(
"""
Model:
- MobileNetV2

Dataset:
- 15 Circuit Symbols

Validation Accuracy:
- 87.97%

Framework:
- TensorFlow + Streamlit
"""
)

@st.cache_resource
def load_model():

    return keras.models.load_model("CircuitSense_MobileNetV2.keras")

model = load_model()

class_names = [
"Ammeter",
"ac_src",
"battery",
"cap",
"curr_src",
"dc_volt_src_1",
"dc_volt_src_2",
"dep_curr_src",
"dep_volt",
"diode",
"gnd_1",
"gnd_2",
"inductor",
"resistor",
"voltmeter"]

uploaded_file = st.file_uploader("Upload a circuit symbol",type=["png","jpg","jpeg","bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image,caption="Uploaded Image",width=300)
    img = image.resize((224,224))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array,0)
    prediction = model.predict(img_array,verbose=0)[0]
    predicted = np.argmax(prediction)
    confidence = prediction[predicted]

    st.subheader("Prediction")
    st.success(class_names[predicted])

    st.subheader("Confidence")

    st.metric("",f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    st.subheader("Top 3 Predictions")

    top3 = np.argsort(prediction)[::-1][:3]

    for idx in top3:
        st.write(
            f"**{class_names[idx]}**: {prediction[idx]*100:.2f}%")