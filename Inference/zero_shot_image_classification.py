"""
Zero-shot image classification is a task that involves classifying images into different categories using a model that
was not explicitly trained on data containing labeled examples from those specific categories.

Traditionally, image classification requires training a model on a specific set of labeled images, and this model learns
 to “map” certain image features to labels. When there’s a need to use such model for a classification task that
 introduces a new set of labels, fine-tuning is required to “recalibrate” the model.

In contrast, zero-shot or open vocabulary image classification models are typically multi-modal models that have been
trained on a large dataset of images and associated descriptions. These models learn aligned vision-language
representations that can be used for many downstream tasks including zero-shot image classification.

This is a more flexible approach to image classification that allows models to generalize to new and unseen categories
without the need for additional training data and enables users to query images with free-form text descriptions of
their target objects .
"""

from transformers import pipeline
from PIL import Image
import requests

checkpoint = "openai/clip-vit-large-patch14"
classifier = pipeline(model=checkpoint, task="zero-shot-image-classification")

url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
image.show()

predictions = classifier(image, candidate_labels=["fox", "bear", "seagull", "owl"])
print(predictions)
