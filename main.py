import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)


class HumanKeypointOnnxModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        if device == "gpu":
            self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        elif device == "cpu":
            self.session = ort.InferenceSession(model_path)
        else:
            raise ValueError(f"Invalid device: {device}")

        self.image_shape = (384, 288)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, self.image_shape[::-1])
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1).reshape(1, 3, self.image_shape[0], self.image_shape[1])
        image = image.astype(np.float32)
        return image

    def predict(self, image: np.ndarray) -> np.ndarray:
        orig_image_shape = image.shape[:2]
        image = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        input_data = {input_name: image}
        output = self.session.run(None, input_data)[0]
        keypoints, maxvals = self.postprocess(output, orig_image_shape)
        return keypoints, maxvals

    def postprocess(self, heatmaps: np.ndarray, orig_image_shape: tuple[int, int], threshold: float = 0.1) -> np.ndarray:
        _, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape

        orig_height, orig_width = orig_image_shape

        keypoints = []
        confidences = []

        for i in range(num_keypoints):
            heatmap = heatmaps[0][i]

            # Find the maximum value (peak) in the heatmap for the keypoint
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)

            confidence = max_val
            if confidence < threshold:
                keypoints.append((None, None))
                confidences.append(0.0)
                continue
            x, y = max_loc

            # Scale the coordinates back to the original image size
            x = int(x * orig_width / heatmap_width)
            y = int(y * orig_height / heatmap_height)

            keypoints.append((x, y))
            confidences.append(confidence)

        return keypoints, confidences