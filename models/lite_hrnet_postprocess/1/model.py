import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def _postprocess(self, heatmaps: np.ndarray, orig_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape

        orig_height, orig_width = orig_shape

        keypoints = []
        confidences = []

        for i in range(num_keypoints):
            heatmap = heatmaps[0][i]

            # Find the maximum value (peak) in the heatmap for the keypoint
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)

            confidence = max_val
            if confidence < 0.1:
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

    def execute(self, requests):
        responses = []
        for request in requests:
            heatmaps = pb_utils.get_input_tensor_by_name(request, "heatmaps")
            heatmaps = heatmaps.as_numpy()

            orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape")
            orig_shape = orig_shape.as_numpy().squeeze()

            keypoints, confidences = self._postprocess(heatmaps, orig_shape)

            keypoints_tensor = pb_utils.Tensor("keypoints", np.array(keypoints, dtype=np.int32))
            confidences_tensor = pb_utils.Tensor("confidences", np.array(confidences, dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[keypoints_tensor, confidences_tensor]))
        return responses
