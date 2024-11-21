import cv2
from src.litehrnet import LiteHRNetClient, LiteHRNetFullClient


def draw_keypoints(image, keypoints, confidences):
    for point, conf in zip(keypoints, confidences):
        if point is not None and conf > 0.1:
            x, y = point
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    return image

def test_lite_hrnet_client_http(url: str):
    client = LiteHRNetClient(url, "lite_hrnet_model", use_http=True)
    keypoints, confidences = client.predict(cv2.imread("tests/images/1.jpg"))
    return keypoints, confidences

def test_lite_hrnet_client_grpc(url: str):
    client = LiteHRNetClient(url, "lite_hrnet_model", use_http=False)
    keypoints, confidences = client.predict(cv2.imread("tests/images/1.jpg"))
    return keypoints, confidences

def test_lite_hrnet_full_client_http(url: str):
    client = LiteHRNetFullClient(url, "lite_hrnet_full", use_http=True)
    keypoints, confidences = client.predict(cv2.imread("tests/images/1.jpg"))
    return keypoints, confidences

def test_lite_hrnet_full_client_grpc(url: str):
    client = LiteHRNetFullClient(url, "lite_hrnet_full", use_http=False)
    keypoints, confidences = client.predict(cv2.imread("tests/images/1.jpg"))
    return keypoints, confidences

if __name__ == "__main__":
    test_lite_hrnet_client_http("localhost:8000")
    test_lite_hrnet_client_grpc("localhost:8001")
    test_lite_hrnet_full_client_http("localhost:8000")
    test_lite_hrnet_full_client_grpc("localhost:8001")