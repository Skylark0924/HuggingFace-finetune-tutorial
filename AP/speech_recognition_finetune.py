"""
Automatic speech recognition (ASR) converts a speech signal to text, mapping a sequence of audio inputs to text outputs.
Virtual assistants like Siri and Alexa use ASR models to help users everyday, and there are many other useful
user-facing applications like live captioning and note-taking during meetings.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset, Audio
from huggingface_hub import login
from transformers import AutoModelForCTC, TrainingArguments, Trainer, AutoProcessor
import evaluate
import rofunc as rf
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Union


def uppercase(example):
    return {"transcription": example["transcription"].upper()}


def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def fine_tune(pre_trained_model, processor, encoded_dataset, dataset_name):
    model = AutoModelForCTC.from_pretrained(
        pre_trained_model, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="speech_rcgn_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=2000,
        gradient_checkpointing=True,
        fp16=True,
        group_by_length=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
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
    dataset_name = "PolyAI/minds14"
    dataset = load_dataset(dataset_name, name="en-US", split="train[:100]")
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.remove_columns(["english_transcription", "intent_class", "lang_id"])
    dataset = dataset.map(uppercase)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # List of supported pre-trained model
    pre_trained_models = ['facebook/wav2vec2-base']

    # load the metric from evaluate
    metric = evaluate.load("wer")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            processor = AutoProcessor.from_pretrained(pre_trained_model)
            encoded_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
            data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, processor, encoded_dataset, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e),
                                  type='warning')
