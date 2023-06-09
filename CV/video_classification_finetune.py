"""
Video classification is the task of assigning a label or class to an entire video. Videos are expected to have only
one class for each video. Video classification models take a video as input and return a prediction about which class
the video belongs to. These models can be used to categorize what a video is all about. A real-world application of
video classification is action / activity recognition, which is useful for fitness applications. It is also helpful for
vision-impaired individuals, especially when they are commuting.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset, Audio
from huggingface_hub import login, hf_hub_download
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
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
    class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique classes: {list(label2id.keys())}.")

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
