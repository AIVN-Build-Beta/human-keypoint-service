
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class LiteHRNetClient:
    def __init__(self, url: str, model_name: str, model_version: int = "1", use_http: bool = False):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version

        if use_http:
            self.client_protocol = httpclient
            self.client = httpclient.InferenceServerClient(url)
        else:
            self.client_protocol = grpcclient
            self.client = grpcclient.InferenceServerClient(url)

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

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        inputs = [self.client_protocol.InferInput("img", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [self.client_protocol.InferRequestedOutput("19302")]
        result = self.client.infer(self.model_name, inputs, outputs=outputs, model_version=self.model_version)
        return result.as_numpy("19302")

    def predict(self, image: np.ndarray) -> np.ndarray:
        orig_image_shape = image.shape[:2]
        preprocessed_image = self.preprocess(image)
        output = self.infer(preprocessed_image)
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


class LiteHRNetFullClient:
    def __init__(self, url: str, model_name: str, model_version: int = "1", use_http: bool = False):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version

        if use_http:
            self.client_protocol = httpclient
            self.client = httpclient.InferenceServerClient(url)
        else:
            self.client_protocol = grpcclient
            self.client = grpcclient.InferenceServerClient(url)

    def infer(self, input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs = [self.client_protocol.InferInput("image", input_data.shape, "UINT8")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [self.client_protocol.InferRequestedOutput("keypoints"), self.client_protocol.InferRequestedOutput("confidences")]
        result = self.client.infer(self.model_name, inputs, outputs=outputs, model_version=self.model_version)
        return result.as_numpy("keypoints"), result.as_numpy("confidences")

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = np.expand_dims(image, axis=0)
        keypoints, confidences = self.infer(image)
        return keypoints, confidences