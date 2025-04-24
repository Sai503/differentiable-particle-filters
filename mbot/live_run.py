from picamera2 import Picamera2
from PIL import Image
import time
picam2 = Picamera2()
picam2.start()
time.sleep(1)
image = picam2.capture_image("main")
arr = picam2.capture_array("main")
print("Image captured")
print(image.size)
print(arr.shape)
# Save the image
image.save("image.jpg")
picam2.stop()