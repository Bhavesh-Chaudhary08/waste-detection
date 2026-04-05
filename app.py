import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
st.set_page_config(page_title="EcoScan AI", layout="wide")
st.title("♻EcoScan: Real-Time Waste Detection")

model_path = r'C:\Users\BHAVESH\PycharmProjects\sign_language\.venv\runs\detect\waste_sorter_final3\weights\best.pt'


@st.cache_resource
def load_model():
    return YOLO(model_path)


model = load_model()

st.sidebar.title("Settings")
mode = st.sidebar.radio("Select Input Mode:", ("Upload Photo", "Live Webcam"))
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)


if mode == "Upload Photo":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)

        results = model.predict(source=img, conf=conf_threshold)
        st.image(results[0].plot(), caption="Detection Result", use_container_width=True)

else:
    st.subheader("Webcam Mode")
    st.write("Click 'Take Photo' to scan the item in front of your camera.")


    img_file = st.camera_input("Scan Trash")

    if img_file:

        bytes_data = img_file.getvalue()
        img = Image.open(io.BytesIO(bytes_data))


        results = model.predict(source=img, conf=conf_threshold)


        st.image(results[0].plot(), caption="AI Scan Result", use_container_width=True)


        count = len(results[0].boxes)
        if count > 0:
            st.success(f" Detected {count} item(s). Please place in the recycling bin.")
        else:
            st.warning("⚠ No trash detected. Adjust lighting or distance.")