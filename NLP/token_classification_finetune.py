import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import evaluate
import numpy as np
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, \
    Trainer
import rofunc as rf


def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, id2label, label2id):
    model = AutoModelForTokenClassification.from_pretrained(
        pre_trained_model, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id
    )

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="token_classification_finetune",
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
    with open('../token.txt') as f:
        my_token = f.read()
    login(my_token)

    dataset = load_dataset("wnut_17")
    label_list = dataset["train"].features[f"ner_tags"].feature.names
    example = dataset["train"][0]

    pre_trained_models = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2',
                          'google/bigbird-roberta-base',
                          'google/bigbird-pegasus-large-arxiv', 'microsoft/biogpt', 'bigscience/bloom-560m',
                          'microsoft/deberta-base', 'microsoft/deberta-v2-xlarge',
                          'bhadresh-savani/electra-base-emotion',
                          'microsoft/DialogRPT-updown', 'jpwahle/longformer-base-plagiarism-detection', 'openai-gpt',
                          'ArthurZ/opt-350m-dummy-sc', 'cardiffnlp/twitter-roberta-base-emotion', 'xlnet-base-cased',
                          'xlm-roberta-xlarge', 'xlm-mlm-en-2048']

    for pre_trained_model in pre_trained_models:
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        seqeval = evaluate.load("seqeval")

        labels = [label_list[i] for i in example[f"ner_tags"]]
        id2label = {
            0: "O",
            1: "B-corporation",
            2: "I-corporation",
            3: "B-creative-work",
            4: "I-creative-work",
            5: "B-group",
            6: "I-group",
            7: "B-location",
            8: "I-location",
            9: "B-person",
            10: "I-person",
            11: "B-product",
            12: "I-product",
        }
        label2id = {
            "O": 0,
            "B-corporation": 1,
            "I-corporation": 2,
            "B-creative-work": 3,
            "I-creative-work": 4,
            "B-group": 5,
            "I-group": 6,
            "B-location": 7,
            "I-location": 8,
            "B-person": 9,
            "I-person": 10,
            "B-product": 11,
            "I-product": 12,
        }

        rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
        fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, id2label, label2id)