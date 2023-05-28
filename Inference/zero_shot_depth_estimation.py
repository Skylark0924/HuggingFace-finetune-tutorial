"""
Monocular depth estimation is a computer vision task that involves predicting the depth information of a scene from a
single image. In other words, it is the process of estimating the distance of objects in a scene from a single camera
viewpoint.

Monocular depth estimation has various applications, including 3D reconstruction, augmented reality, autonomous driving,
and robotics. It is a challenging task as it requires the model to understand the complex relationships between objects
in the scene and the corresponding depth information, which can be affected by factors such as lighting conditions,
occlusion, and texture.
"""

from transformers import pipeline
from PIL import Image
import requests

pre_trained_model = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=pre_trained_model)

url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
image.show()

predictions = depth_estimator(image)
predictions["depth"].show()
