from locust import HttpUser, TaskSet, task, between
import os

class UserBehavior(TaskSet):
    @task
    def upload_image(self):
        # Ganti dengan nama file gambar yang ingin Anda uji
        image_file_name = '1.jpeg'  # Ganti dengan nama file yang sesuai
        image_path = os.path.join('images', image_file_name)  # Ganti dengan path yang sesuai

        with open(image_path, 'rb') as file:
            response = self.client.post(
                "/calorie", 
                files={"image": file},
                verify=False  # Nonaktifkan verifikasi SSL (hanya untuk pengujian, tidak disarankan untuk produksi)
            )
            print(f"Uploaded {image_file_name}: {response.json()}")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(0, 7200)  # Rata-rata 1 permintaan setiap 1,2 menit (7200 detik)

    def on_start(self):
        """ Called when a simulated user starts running """
        self.client.verify = False  # Nonaktifkan verifikasi SSL (hanya untuk pengujian)

