"""
Question answering tasks return an answer given a question. If you’ve ever asked a virtual assistant like Alexa, Siri
or Google what the weather is, then you’ve used a question answering model before. There are two common types
of question answering tasks:
- Extractive: extract the answer from the given context.
- Abstractive: generate an answer from the context that correctly answers the question.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, \
    Trainer
import rofunc as rf


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator):
    model = AutoModelForQuestionAnswering.from_pretrained(pre_trained_model)

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="QA_{}_finetune".format(pre_trained_model),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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

    dataset = load_dataset("squad", split="train[:5000]")
    dataset = dataset.train_test_split(test_size=0.2)

    pre_trained_models = ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2',
                          'google/bigbird-roberta-base',
                          'google/bigbird-pegasus-large-arxiv', 'microsoft/biogpt', 'bigscience/bloom-560m',
                          'microsoft/deberta-base', 'microsoft/deberta-v2-xlarge',
                          'bhadresh-savani/electra-base-emotion',
                          'microsoft/DialogRPT-updown', 'jpwahle/longformer-base-plagiarism-detection', 'openai-gpt',
                          'ArthurZ/opt-350m-dummy-sc', 'cardiffnlp/twitter-roberta-base-emotion', 'xlnet-base-cased',
                          'xlm-roberta-xlarge', 'xlm-mlm-en-2048']

    for pre_trained_model in pre_trained_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
            tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
            data_collator = DefaultDataCollator()

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, tokenizer, tokenized_dataset, data_collator)
        except:
            rf.utils.beauty_print('Model {} is not suitable'.format(pre_trained_model), type='warning')
