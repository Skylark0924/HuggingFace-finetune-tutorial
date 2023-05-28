"""
Object detection is the computer vision task of detecting instances (such as humans, buildings, or cars) in an image.
Object detection models receive an image as input and output coordinates of the bounding boxes and associated labels
of the detected objects. An image can contain multiple objects, each with its own bounding box and a label (e.g. it
can have a car and a building), and each object can be present in different parts of an image (e.g. the image can have
several cars). This task is commonly used in autonomous driving for detecting things like pedestrians, road signs, and
traffic lights. Other applications include counting objects in images, image search, and more.
"""

# TODO

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
import rofunc as rf
import albumentations
import numpy as np


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)
    return annotations


# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]
    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def fine_tune(pre_trained_model, image_processor, encoded_dataset, label2id, id2label, dataset_name):
    model = AutoModelForObjectDetection.from_pretrained(
        pre_trained_model,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="obj_detect_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        per_device_train_batch_size=8,
        num_train_epochs=10,
        fp16=True,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=encoded_dataset["train"],
        tokenizer=image_processor,
    )

    # Start fine-tuning
    trainer.train()

    # Share your trained model to Hub
    trainer.push_to_hub()


if __name__ == '__main__':
    shutup.please()

    # Login in with your own API token
    with open('../token.txt') as f:
        my_token = f.read()
    login(my_token)

    # Load the dataset
    dataset_name = "cppe-5"
    dataset = load_dataset(dataset_name)
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(dataset["train"])) if i not in remove_idx]
    dataset["train"] = dataset["train"].select(keep)
    dataset["train"] = dataset["train"].with_transform(transform_aug_ann)
    encoded_dataset = dataset

    # List of supported pre-trained model
    pre_trained_models = ['facebook/detr-resnet-50', 'hustvl/yolos-tiny']

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            image_processor = AutoImageProcessor.from_pretrained(pre_trained_model)
            transform = albumentations.Compose(
                [
                    albumentations.Resize(480, 480),
                    albumentations.HorizontalFlip(p=1.0),
                    albumentations.RandomBrightnessContrast(p=1.0),
                ],
                bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
            )

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, image_processor, encoded_dataset, label2id, id2label, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e),
                                  type='warning')
