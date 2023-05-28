"""
Semantic segmentation assigns a label or class to each individual pixel of an image. There are several types of
segmentation, and in the case of semantic segmentation, no distinction is made between unique instances of the same
object. Both objects are given the same label (for example, “car” instead of “car-1” and “car-2”). Common real-world
applications of semantic segmentation include training self-driving cars to identify pedestrians and important traffic
information, identifying cells and abnormalities in medical imagery, and monitoring environmental changes from satellite
imagery.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, TrainingArguments, Trainer
import evaluate
import rofunc as rf
import numpy as np
import json
import torch
from torch import nn
from huggingface_hub import cached_download, hf_hub_url
from torchvision.transforms import ColorJitter


def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics


def fine_tune(pre_trained_model, train_ds, test_ds, label2id, id2label, dataset_name):
    model = AutoModelForSemanticSegmentation.from_pretrained(
        pre_trained_model, label2id=label2id, id2label=id2label
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="semantic_seg_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        learning_rate=6e-5,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
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
    dataset_name = "scene_parse_150"
    dataset = load_dataset(dataset_name, split="train[:50]")
    dataset = dataset.train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    repo_id = "huggingface/label-files"
    filename = "ade20k-hf-doc-builder.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    # List of supported pre-trained model
    pre_trained_models = ['nvidia/mit-b0']

    # load the metric from evaluate
    metric = evaluate.load("mean_iou")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            image_processor = AutoImageProcessor.from_pretrained(pre_trained_model, reduce_labels=True)
            jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, train_ds, test_ds, label2id, id2label, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e),
                                  type='warning')
