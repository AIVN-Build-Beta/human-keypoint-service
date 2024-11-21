import io
import random

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


class ImageGenerator:
    @staticmethod
    def generate_random_image(width=224, height=224):
        random_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(random_image)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr


class PredictAPIUser(HttpUser):
    wait_time = between(1, 3)  # Random wait between requests

    def on_start(self):
        self.image_generator = ImageGenerator()

    @task
    def predict_image(self):
        image_height = random.randint(512, 1024)
        image_width = random.randint(512, 1024)
        image_data = self.image_generator.generate_random_image(image_width, image_height)

        files = {
            'file': ('image.png', image_data, 'image/png')
        }

        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Request failed with status code: {response.status_code}")