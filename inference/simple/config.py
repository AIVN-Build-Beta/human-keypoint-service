from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    weight_path: str = "models/model.onnx"
    device: Literal["cpu", "gpu"] = "gpu"
    confidence_threshold: float = 0.1
    max_image_size: int = 1920  # Maximum size for any dimension

    class Config:
        env_prefix = "KEYPOINT_"  # Environment variables will be prefixed with KEYPOINT_