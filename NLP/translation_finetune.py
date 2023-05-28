"""
Translation converts a sequence of text from one language to another. It is one of several tasks you can formulate as
a sequence-to-sequence problem, a powerful framework for returning some output from an input, like translation or
summarization. Translation systems are commonly used for translation between different language texts, but it can also
be used for speech or some combination in between like text-to-speech or speech-to-text.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
import evaluate
import rofunc as rf
import numpy as np

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def fine_tune(pre_trained_model, tokenizer, lm_dataset, data_collator, dataset_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(pre_trained_model)

    # Define training hyperparameters
    training_args = Seq2SeqTrainingArguments(
        output_dir="translation_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Seq2SeqTrainer(
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
    dataset_name = "opus_books"
    dataset = load_dataset(dataset_name, "en-fr")
    dataset = dataset["train"].train_test_split(test_size=0.2)

    # List of supported pre-trained model
    pre_trained_models = ['t5-small']

    # load the metric from evaluate
    metric = evaluate.load("sacrebleu")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=dataset["train"].column_names,
            )
            tokenizer.pad_token = tokenizer.eos_token
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=pre_trained_model)

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator, dataset_name)
        except:
            rf.utils.beauty_print('Model {} is not suitable'.format(pre_trained_model), type='warning')
