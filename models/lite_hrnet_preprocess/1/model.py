import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.image_shape = (384, 288)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _normalize_image(self, image):
        image = cv2.resize(image, self.image_shape[::-1])
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1).reshape(1, 3, self.image_shape[0], self.image_shape[1])
        image = image.astype(np.float32)
        return image

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "image").as_numpy().squeeze()
            input_shape = np.array(input_tensor.shape[:2], dtype=np.int32).reshape(1, 2)
            orig_shape_tensor = pb_utils.Tensor("orig_shape", input_shape)

            preprocessed_image = self._normalize_image(input_tensor)
            output_tensor = pb_utils.Tensor("preprocessed_image", preprocessed_image)

            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor, orig_shape_tensor]))
        return responses