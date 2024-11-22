import logging

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from src.config import app_config
from src.litehrnet import LiteHRNetClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"App config: {app_config}")

# Set the page layout to wide
st.set_page_config(layout="wide")

client = LiteHRNetClient(
    url=app_config.triton_url,
    model_name=app_config.model_name,
    model_version=app_config.model_version,
    use_http=app_config.use_http
)

def detect_keypoints(image):
    keypoints, conf = client.predict(image)
    logger.info("Keypoints detected")
    return keypoints

def plot_keypoints(image, keypoints):
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for point in keypoints:
        if point[0] is None or point[1] is None:
            continue
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_cv, (x, y), 5, (0, 255, 0), -1)
    return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

def push_metrics(image_count, detection_count):
    registry = CollectorRegistry()
    image_upload_gauge = Gauge('image_uploads', 'Number of images uploaded', registry=registry)
    keypoint_detection_gauge = Gauge('keypoint_detections', 'Number of keypoint detections performed', registry=registry)

    image_upload_gauge.set(image_count)
    keypoint_detection_gauge.set(detection_count)

    push_to_gateway(app_config.pushgateway_url, job='streamlit_app', registry=registry)

st.title("Human Keypoints Detection App")

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    image_count = len(uploaded_files)
    logger.info(f"{image_count} images uploaded")

    columns = st.columns(app_config.num_columns)
    detection_count = 0

    for idx, uploaded_file in enumerate(uploaded_files):
        col = columns[idx % app_config.num_columns]
        with col:
            image = Image.open(uploaded_file)
            thumbnail = image.resize(app_config.thumbnail_size)
            st.image(thumbnail, use_container_width=True)

            image_np = np.array(image)
            keypoints = detect_keypoints(image_np)
            detection_count += len(keypoints)
            keypoints_image = plot_keypoints(image_np, keypoints)

            with st.expander("View"):
                st.image(image, caption="Original Image", use_container_width=True)
                st.image(keypoints_image, caption="Processed Image", use_container_width=True)

    push_metrics(image_count, detection_count)