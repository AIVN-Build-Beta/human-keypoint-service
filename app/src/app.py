import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.config import app_config
from src.litehrnet import LiteHRNetClient

# Set the page layout to wide
st.set_page_config(layout="wide")

# Initialize the LiteHRNetClient with configuration
client = LiteHRNetClient(url=app_config.triton_url,
                         model_name=app_config.model_name,
                         model_version=app_config.model_version,
                         use_http=app_config.use_http)

def detect_keypoints(image):
    # Use LiteHRNetClient to detect keypoints
    keypoints, conf = client.predict(image)
    return keypoints

def plot_keypoints(image, keypoints):
    # Convert the image to a format suitable for OpenCV
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw keypoints on the image
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_cv, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle for each keypoint

    # Convert back to RGB for display in Streamlit
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return image_rgb

st.title("Human Keypoints Detection App")

# Upload multiple images
uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Use the number of columns from config
    num_columns = app_config.num_columns
    columns = st.columns(num_columns)

    for idx, uploaded_file in enumerate(uploaded_files):
        # Determine which column to use
        col = columns[idx % num_columns]

        with col:
            # Display thumbnail without caption
            image = Image.open(uploaded_file)
            thumbnail_size = app_config.thumbnail_size  # Use thumbnail size from config
            thumbnail = image.resize(thumbnail_size)  # Resize to thumbnail size
            st.image(thumbnail, use_container_width=True)  # No caption

            # Convert image to numpy array for processing
            image_np = np.array(image)

            # Process image with keypoints detection model
            keypoints = detect_keypoints(image_np)

            # Plot keypoints on the image
            keypoints_image = plot_keypoints(image_np, keypoints)

            # Use expander to show original and processed images
            with st.expander(f"View"):
                st.image(image, caption="Original Image", use_container_width=True)
                st.image(keypoints_image, caption="Processed Image", use_container_width=True)