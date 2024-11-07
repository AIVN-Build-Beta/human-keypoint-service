from main import HumanKeypointOnnxModel
import cv2


if __name__ == "__main__":
    model = HumanKeypointOnnxModel("models/model.onnx", "gpu")
    keypoints, confidences = model.predict(cv2.imread("tests/images/1.jpg"))

    image = cv2.imread("tests/images/1.jpg")
    for point, conf in zip(keypoints, confidences):
        if point is not None and conf > 0.1:
            x, y = point
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    output_path = "1_with_keypoints.jpg"
    cv2.imwrite(output_path, image)
