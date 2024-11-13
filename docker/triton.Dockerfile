FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN apt update && apt install -y libopencv-dev libopencv-core-dev libgl1-mesa-glx

RUN pip install opencv-python numpy

# Copy model config and model files
COPY models/ /models

# Start Triton server when container launches
CMD ["tritonserver", "--model-repository=/models"]