import logging
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from prometheus_client import Counter, start_http_server
from src.config import app_config
from src.litehrnet import LiteHRNetClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_http_server(app_config.prometheus_port)

image_upload_counter = Counter('image_uploads', 'Number of images uploaded')
keypoint_detection_counter = Counter('keypoint_detections', 'Number of keypoint detections performed')

st.set_page_config(layout="wide")

client = LiteHRNetClient(
    url=app_config.triton_url,
    model_name=app_config.model_name,
    model_version=app_config.model_version,
    use_http=app_config.use_http
)

def detect_keypoints(image):
    keypoints, conf = client.predict(image)
    keypoint_detection_counter.inc()
    logger.info("Keypoints detected")
    return keypoints

def plot_keypoints(image, keypoints):
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_cv, (x, y), 5, (0, 255, 0), -1)
    return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

st.title("Human Keypoints Detection App")

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    image_upload_counter.inc(len(uploaded_files))
    logger.info(f"{len(uploaded_files)} images uploaded")

    columns = st.columns(app_config.num_columns)

    for idx, uploaded_file in enumerate(uploaded_files):
        col = columns[idx % app_config.num_columns]
        with col:
            image = Image.open(uploaded_file)
            thumbnail = image.resize(app_config.thumbnail_size)
            st.image(thumbnail, use_container_width=True)

            image_np = np.array(image)
            keypoints = detect_keypoints(image_np)
            keypoints_image = plot_keypoints(image_np, keypoints)

            with st.expander("View"):
                st.image(image, caption="Original Image", use_container_width=True)
                st.image(keypoints_image, caption="Processed Image", use_container_width=True)