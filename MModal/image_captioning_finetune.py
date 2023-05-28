"""
Image captioning is the task of predicting a caption for a given image. Common real world applications of it include
aiding visually impaired people that can help them navigate through different situations. Therefore, image captioning
helps to improve content accessibility for people by describing images to them.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutup
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer
import evaluate
import rofunc as rf


def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = metric.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}


def fine_tune(pre_trained_model, train_ds, test_ds, dataset_name):
    model = AutoModelForCausalLM.from_pretrained(pre_trained_model)

    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="image_caption_{}_{}_finetune".format(pre_trained_model.split('/')[-1], dataset_name.split('/')[-1]),
        learning_rate=5e-5,
        num_train_epochs=50,
        fp16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=True,
        label_names=["labels"],
        load_best_model_at_end=True,
    )

    # Pass arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
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
    dataset_name = "lambdalabs/pokemon-blip-captions"
    dataset = load_dataset(dataset_name)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    # List of supported pre-trained model
    pre_trained_models = ['microsoft/git-base']

    # load the metric from evaluate
    metric = evaluate.load("wer")

    for pre_trained_model in pre_trained_models:
        try:
            # preprocess the dataset
            processor = AutoProcessor.from_pretrained(pre_trained_model)
            train_ds.set_transform(transforms)
            test_ds.set_transform(transforms)

            rf.utils.beauty_print('Current model: {}'.format(pre_trained_model), type='module')
            fine_tune(pre_trained_model, train_ds, test_ds, dataset_name)
        except Exception as e:
            rf.utils.beauty_print('Model {} is not suitable, Exception: {}'.format(pre_trained_model, e),
                                  type='warning')
