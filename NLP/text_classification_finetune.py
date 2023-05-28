"""
Text classification is a common NLP task that assigns a label or class to text. Some of the largest companies run text
classification in production for a wide range of practical applications. One of the most popular forms of text
classification is sentiment analysis, which assigns a label like 🙂 positive, 🙁 negative, or 😐 neutral to a
sequence of text.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
import numpy as np
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
import rofunc as rf


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, id2label, label2id, dataset_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_trained_model, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="text_cls_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
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
    dataset_name = "imdb"
    dataset = load_dataset(dataset_name)

    # List of supported pre-trained model
    pre_trained_models = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2',
                          'google/bigbird-roberta-base',
                          'google/bigbird-pegasus-large-arxiv', 'microsoft/biogpt', 'bigscience/bloom-560m',
                          'microsoft/deberta-base', 'microsoft/deberta-v2-xlarge',
                          'bhadresh-savani/electra-base-emotion',
                          'microsoft/DialogRPT-updown', 'jpwahle/longformer-base-plagiarism-detection', 'openai-gpt',
                          'ArthurZ/opt-350m-dummy-sc', 'cardiffnlp/twitter-roberta-base-emotion', 'xlnet-base-cased',
                          'xlm-roberta-xlarge', 'xlm-mlm-en-2048']

    # load the metric from evaluate
    metric = evaluate.load("accuracy")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            #  create a map of the expected ids to their labels with id2label and label2id:
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            label2id = {"NEGATIVE": 0, "POSITIVE": 1}

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, id2label, label2id, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e), type='warning')
