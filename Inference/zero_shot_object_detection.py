"""
Traditionally, models used for object detection require labeled image datasets for training, and are limited to
detecting the set of classes from the training data.

Zero-shot object detection is supported by the OWL-ViT model which uses a different approach. OWL-ViT is an
open-vocabulary object detector. It means that it can detect objects in images based on free-text queries without
the need to fine-tune the model on labeled datasets.

OWL-ViT leverages multi-modal representations to perform open-vocabulary detection. It combines CLIP with lightweight
object classification and localization heads. Open-vocabulary detection is achieved by embedding free-text queries with
the text encoder of CLIP and using them as input to the object classification and localization heads. associate images
and their corresponding textual descriptions, and ViT processes image patches as inputs. The authors of OWL-ViT first
trained CLIP from scratch and then fine-tuned OWL-ViT end to end on standard object detection datasets using a bipartite
matching loss.

With this approach, the model can detect objects based on textual descriptions without prior training on labeled
datasets.
"""

import numpy as np
import skimage
from PIL import Image, ImageDraw
from transformers import pipeline

pre_trained_model = "google/owlvit-base-patch32"
detector = pipeline(model=pre_trained_model, task="zero-shot-object-detection")

image = skimage.data.astronaut()
image = Image.fromarray(np.uint8(image)).convert("RGB")
predictions = detector(
    image,
    candidate_labels=["human hair", "human face", 'human eye', "rocket", "nasa badge", "star-spangled banner"],
)

draw = ImageDraw.Draw(image)

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]

    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

image.show()
