"""
Audio classification - just like with text - assigns a class label output from the input data. The only difference is
instead of text inputs, you have raw audio waveforms. Some practical applications of audio classification include
identifying speaker intent, language classification, and even animal species by their sounds.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset, Audio
from huggingface_hub import login
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import rofunc as rf
import numpy as np


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def fine_tune(pre_trained_model, feature_extractor, encoded_dataset, label2id, id2label, dataset_name):
    model = AutoModelForAudioClassification.from_pretrained(
        pre_trained_model, num_labels=len(id2label), label2id=label2id, id2label=id2label
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="audio_cls_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
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
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
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
    dataset = load_dataset(dataset_name, name="en-US", split="train")
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
    labels = dataset["train"].features["intent_class"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # List of supported pre-trained model
    pre_trained_models = ['facebook/wav2vec2-base',
                          'patrickvonplaten/wavlm-libri-clean-100h-base-plus',
                          'patrickvonplaten/unispeech-large-1500h-cv-timit',
                          'microsoft/unispeech-sat-base-100h-libri-ft']

    # load the metric from evaluate
    metric = evaluate.load("accuracy")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            feature_extractor = AutoFeatureExtractor.from_pretrained(pre_trained_model)
            encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
            encoded_dataset = encoded_dataset.rename_column("intent_class", "label")

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, feature_extractor, encoded_dataset, label2id, id2label, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e),
                                  type='warning')
