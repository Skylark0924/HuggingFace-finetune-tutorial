"""
Image classification assigns a label or class to an image. Unlike text or audio classification, the inputs are the
pixel values that comprise an image. There are many applications for image classification, such as detecting damage
after a natural disaster, monitoring crop health, or helping screen medical images for signs of disease.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset, Audio
from huggingface_hub import login
from transformers import AutoModelForImageClassification, AutoImageProcessor, DefaultDataCollator, TrainingArguments, \
    Trainer
import evaluate
import rofunc as rf
import numpy as np
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor


def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune(pre_trained_model, image_processor, encoded_dataset, label2id, id2label, dataset_name):
    model = AutoModelForImageClassification.from_pretrained(
        pre_trained_model, num_labels=len(id2label), label2id=label2id, id2label=id2label
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="image_cls_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=image_processor,
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
    dataset_name = "food101"
    dataset = load_dataset(dataset_name, split="train[:5000]")
    dataset = dataset.train_test_split(test_size=0.2)
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # List of supported pre-trained model
    pre_trained_models = ['google/vit-base-patch16-224-in21k', 'microsoft/beit-base-patch16-224', 'google/bit-50',
                          'facebook/convnext-tiny-224', 'facebook/convnextv2-tiny-1k-224', 'microsoft/cvt-13',
                          ]

    # load the metric from evaluate
    metric = evaluate.load("accuracy")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            image_processor = AutoImageProcessor.from_pretrained(pre_trained_model)
            normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
            size = (
                image_processor.size["shortest_edge"]
                if "shortest_edge" in image_processor.size
                else (image_processor.size["height"], image_processor.size["width"])
            )
            _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
            encoded_dataset = dataset.with_transform(transforms)
            data_collator = DefaultDataCollator()

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, image_processor, encoded_dataset, label2id, id2label, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e),
                                  type='warning')
