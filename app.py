from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import cv2
import numpy as np
from config import Settings
from fastapi import FastAPI, HTTPException, UploadFile
from main import HumanKeypointOnnxModel
from pydantic import BaseModel

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    settings = Settings()
    model = HumanKeypointOnnxModel(settings.model_path, settings.device)
    yield


app = FastAPI(title="Human Keypoint Detection API", lifespan=lifespan)
settings = Settings()


class KeypointResponse(BaseModel):
    keypoints: List[Optional[Tuple[int, int]]]
    confidences: List[float]


@app.post("/predict", response_model=KeypointResponse)
async def predict_keypoints(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Resize if image is too large
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim > settings.max_image_size:
        scale = settings.max_image_size / max_dim
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)

    keypoints, confidences = model.predict(image)

    keypoints = [(int(x), int(y)) if x is not None else None for x, y in keypoints]
    confidences = [float(conf) for conf in confidences]
    return KeypointResponse(keypoints=keypoints, confidences=confidences)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)