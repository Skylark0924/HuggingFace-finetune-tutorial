"""
Masked language modeling predicts a masked token in a sequence, and the model can attend to tokens bidirectionally.
This means the model has full access to the tokens on the left and right. Masked language modeling is great for tasks
that require a good contextual understanding of an entire sequence. BERT is an example of a masked language model.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, \
    Trainer
import rofunc as rf


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def fine_tune(pre_trained_model, lm_dataset, data_collator, dataset_name):
    model = AutoModelForMaskedLM.from_pretrained(pre_trained_model)

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="masked_lm_{}_{}_finetune".format(pre_trained_model, dataset_name),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
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

    dataset_name = "eli5"
    dataset = load_dataset(dataset_name, split="train_asks[:5000]")
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.flatten()

    pre_trained_models = ['distilroberta-base']

    for pre_trained_model in pre_trained_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=dataset["train"].column_names,
            )
            lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
            tokenizer.pad_token = tokenizer.eos_token
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, lm_dataset, data_collator, dataset_name)
        except:
            rf.utils.beauty_print('Model {} is not suitable'.format(pre_trained_model), type='warning')
