import evaluate
import numpy as np
import shutup
from datasets import load_dataset
from huggingface_hub import interpreter_login, notebook_login, login
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
import rofunc as rf


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, id2label, label2id):
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_trained_model, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="text_classification_finetune",
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
    with open('../token/token.txt') as f:
        my_token = f.read()
    login(my_token)

    # Load the dataset
    dataset = load_dataset("imdb")

    # List of supported pre-trained model
    pre_trained_models = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2',
                          'google/bigbird-roberta-base',
                          'google/bigbird-pegasus-large-arxiv', 'microsoft/biogpt', 'bigscience/bloom-560m',
                          'microsoft/deberta-base', 'microsoft/deberta-v2-xlarge',
                          'bhadresh-savani/electra-base-emotion',
                          'microsoft/DialogRPT-updown', 'jpwahle/longformer-base-plagiarism-detection', 'openai-gpt',
                          'ArthurZ/opt-350m-dummy-sc', 'cardiffnlp/twitter-roberta-base-emotion', 'xlnet-base-cased',
                          'xlm-roberta-xlarge', 'xlm-mlm-en-2048']

    for pre_trained_model in pre_trained_models:
        # preprocess the datasetd
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, use_fast=False)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # load the accuracy metric from evaluate
        accuracy = evaluate.load("accuracy")

        #  create a map of the expected ids to their labels with id2label and label2id:
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
        fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, id2label, label2id)
