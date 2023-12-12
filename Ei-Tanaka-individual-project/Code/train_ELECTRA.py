"""
Author: Ei Tanaka
Date: Nov 28, 2023
Purpose: Train ELECTRA model
References:
"""

# ====================================== Import ======================================
import argparse
import os
import sys
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, load_metric, DatasetDict

from accelerate import Accelerator
from tqdm.auto import tqdm
import collections

# ====================================== Setup ======================================
OS_PATH = os.getcwd()
os.chdir("..")
ROOT_PATH = os.getcwd()
MODEL_PATH = os.path.join(os.getcwd(), "Models", "ELECTRA")
os.chdir(OS_PATH)

model_name = 'ELECTRA-finetuned-squadv2-accelerate'

# Random seed
manual_seed = 42
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ====================================== Hyperparameters ======================================
max_length = 384    # Max length of input sentence
stride = 128    # Max length of input sentence
n_best = 20    # Number of predictions to generate
max_answer_length = 30    # Max length of answer

model_checkpoint = "google/electra-base-discriminator"
tokenizer = ElectraTokenizerFast.from_pretrained(model_checkpoint)

learning_rate = 5e-5
batch_size = 6
num_epochs = 3

# ====================================== Function ======================================
"""
The hard part will be to generate labels for the question's answer, which will be the start and end positions of the answer in the context.
1. Converts the text in the input into IDs the model can make sense of using the tokenizer. (ElectraTokenizerFast)
The labels will then be the index of the tokens staring and ending the answer, and the model will be tasked to predicted one start and end logit per token in the input.
2. Create the sliding window of length max_length with a stride of stride. Based on the max length of the input, the context and question will be split into multiple windows.
3. Pad the inputs and labels to the max_length.
4. Generates offsets mapping the tokens to their position in the context. (offset_mapping) If the token is either start or end of the answer, the offset will be 1.
5. Overflow_to_sample_mapping is a mapping between the index of the window and the index of the original sample. Ex. if the text 1 is devided into 3 and texxt 2 id devided into 4 text,  the mapping will be [0, 0, 0, 1, 1, 1, 1].
"""
def preprocess_training_examples(examples):
    """
    Truncate the context and question to max_length and return the start and end positions in the original context.
    :param examples: (dict) dictionary of training examples
    :return: (dict) dictionary of preprocessed training examples
    """

    # Tokenize questions and context
    questions = [q.strip() for q in examples["question"]]   # Remove leading and trailing whitespaces

    # Tokenize questions and context
    inputs = tokenizer(
        questions,  # List of questions to be encoded as a single input
        examples["context"],    # List of contexts to be encoded as a single input
        max_length=max_length,  # Max length of input sentence
        truncation="only_second",   # Only truncate the context
        stride=stride,  # Pad to max_length
        return_overflowing_tokens=True,   # Return all of the overflowing tokens
        return_offsets_mapping=True,    # Return the mapping between tokens and character positions
        padding="max_length",   # Pad to max_length
    )

    offset_mapping = inputs.pop("offset_mapping")   # Remove offset_mapping from inputs
    sample_map = inputs.pop("overflow_to_sample_mapping")   # Remove overflow_to_sample_mapping from inputs
    answers = examples["answers"]   # Answers
    start_positions = []    # Start positions
    end_positions = []  # End positions

    # For each sample in the batch (or for each batch, depending on the batch size)
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]  # Get the sample index for the current batch index
        answer = answers[sample_idx]    # Get the answer for the current sample index
        if len(answer["answer_start"]) == 0:    # If the answer is empty, the answer is (0, 0)
            start_char = 0
            end_char = 0
        else:
            start_char = answer["answer_start"][0]   # Get the start character of the answer
            end_char = answer['answer_start'][0] + len(answer["text"][0])  # Get the end character of the answer
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
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


def preprocessing_validation_example(examples):
    """
    Preprocess the validation examples by truncating the context and question to max_length and return the start and end positions in the original context.
    :param examples: (dict) dictionary of validation examples
    :return: (dict) dictionary of preprocessed validation examples
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids

    return inputs

def process_data_to_model_inputs(train_dataset, validation_dataset):
    """ Tokenize the inputs and labels for the model.
    :param train_dataset:  the training dataset
    :param validation_dataset:  the validation dataset
    :return: train_dataloader, eval_dataloader
    """
    # Tokenize the inputs and labels for the model.
    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size,
    )

    eval_dataloader = DataLoader(
        validation_set,
        collate_fn=default_data_collator,
        batch_size=batch_size,
    )

    return train_dataloader, eval_dataloader

# ====================================== Model ======================================
"""
Fine-tune the model on the SQuAD dataset with the Trainer API.
The model will output the start and end logits which will be used to compute the loss.
Steps:
1. We masked the start and end logits corresponding to tokens outside of the context.
2. We then converted the start and end logits into probabilities using the softmax function.
3. We attributed a score to each pair by taking the product of the start and end probabilities.
4. We looked for the pair with the maximum score that yielded a valid answer.
"""
def model_init():
    """
    Initialize the model.
    :return: (AutoModelForQuestionAnswering) the model
    """
    model = ElectraForQuestionAnswering.from_pretrained(model_checkpoint)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer

# ====================================== Training ======================================
def train(train_dataloader, eval_dataloader, model, optimizer, accelerator, validation_dataset, raw_dataset):

    num_train_epochs = num_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs

    lr_shceduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(num_train_epochs):
        with tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch + 1}/{num_train_epochs}", unit="batch") as progress_bar:

            model.train()

            for step, batch in enumerate(train_dataloader):

                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_shceduler.step()
                progress_bar.update(1)
                optimizer.zero_grad()

            # Evaluation
            model.eval()
            start_logits = []
            end_logits = []
            print('\n')

        accelerator.print(f"***** Running evaluation *****")
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)     # (num_eval_examples, context_len)
        end_logits = np.concatenate(end_logits)     # (num_eval_examples, context_len)

        start_logits = start_logits[:, len(validation_dataset)]    # (num_eval_examples, )
        end_logits = end_logits[:, len(validation_dataset)]     # (num_eval_examples, )

        # Compute the validation metrics
        mertics = compute_metrics(
            start_logits, end_logits, validation_dataset, raw_dataset["validation"]
        )

        print(f"Epoch: {epoch}", mertics)
        print('*' * 50)

        # Save the model checkpoint
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(MODEL_PATH, save_function=accelerator.save)
        if accelerator.is_local_main_process:
            tokenizer.save_pretrained(MODEL_PATH)

# ====================================== Fine Tuning ======================================
def fine_tune(model, train_dataset, eval_dataset, raw_datasets):
    """
    Fine-tune the model on the SQuAD dataset with the Trainer API.
    :param model:
    :param train_dataset:
    :param eval_dataset:
    :return:
    """

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_PATH, "electra-finetuned-squadv2"),
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(os.path.join(MODEL_PATH, "electra-finetuned-squadv2"))

    predictions, labels, metrics = trainer.predict(eval_dataset)
    start_logits, end_logits = predictions

    return compute_metrics(start_logits, end_logits, eval_dataset, raw_datasets["validation"])

# ====================================== Evaluation ======================================
def evaluate(model, eval_dataloader, eval_dataset, raw_datasets, accelerator):

    model.eval()
    start_logits = []
    end_logits = []

    with tqdm(total=len(eval_dataloader), desc="Evaluating", unit="batch") as progress_bar:
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            progress_bar.update(1)

        start_logits = np.concatenate(start_logits)  # (num_eval_examples, context_len)
        end_logits = np.concatenate(end_logits)  # (num_eval_examples, context_len)

        start_logits = start_logits[:, len(eval_dataloader)]  # (num_eval_examples, )
        end_logits = end_logits[:, len(eval_dataloader)]  # (num_eval_examples, )

    # Compute the validation metrics
    mertics = compute_metrics(
        start_logits, end_logits, eval_dataset, raw_datasets["validation"]
    )

    return mertics

# ====================================== Prediction ======================================
def predict(model_path, question, context):
    """
    Predict the answer to a question given the context.
    :param model:
    :param question:
    :param context:
    :return:
    """

    model_path = model_path
    model = ElectraForQuestionAnswering.from_pretrained(model_path)

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs).to_tuple()

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# ====================================== Compute Metrics ======================================
"""
The hardest part of the pipeline is to compute the metrics.
Since we padded all the samples to the maximum length we set, there is no data collator to define, so this metric computation is really specific to the task.
The difficult part will be to post-processing the model predictions into spans of text that can be compared to the ground truth.
Steps:
We need see how we will get one answer for each example in our validation set.
The processing of the initial dataset implies splitting examples in several features, which may or may not contain the answer.
Passing those features through the model will yield several predictions for each feature (logit).
The model will give us logits for start and end positions for each feature, since our labels are the indices of the tokens that correspond to the answer.
We must then somehow convert those logits into an answer, and then, pick one of the various answers for each feature gives to be the answer for the example.

1. We will need the map from examples to features, which we can create by looping through all the features and storing the example_id associated to each feature.
2. We could just take the best index for the start and end logits and be done. But if our model predict something impossible, like the token in the quesiton, 
we will look at more of the logits. We attribute a score to each pair of start and end logits by taking the product of their probabilities.
3. We loop through the best start and end logits and pick the corresponding answers.
4. We can check our predicted answer with the label.
5. We can then loop through all the possible examples.

"""
def compute_metrics(start_logits, end_logits, features, examples):
    example_to_feature = collections.defaultdict(list)
    # Loop through all the features to gather the mapping from example to features.
    for idx, feature in enumerate(features):
        example_to_feature[feature["example_id"]].append(idx)

    predicted_answers = []
    theoretical_answers = []

    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all the features associated to the example_id in order to gather all the predictions
        for feature_idx in example_to_feature[example_id]:

            # We grave the actual answer for this feature
            theoretical_answers.append(
                {
                    "id": example_id,
                    "answers": example["answers"],
                }
            )

            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offsets = features[feature_idx]["offset_mapping"]

            # Gather the indices the best start/end logits
            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:

                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]:offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

            # Select the answer with the highest `start_logit + end_logit` score as the final answer
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {
                        "id": example_id,
                        "prediction_text": best_answer["text"],
                        "no_answer_probability": 0.0,
                    }
                )
            else:
                predicted_answers.append(
                    {
                        "id": example_id,
                        "prediction_text": "",
                        "no_answer_probability": 1.0,
                    }
                )

    metric = load_metric("squad_v2")

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

# ====================================== Main ======================================
def main():
    # Load dataset
    raw_dataset = load_dataset("squad_v2")

    # Preprocess dataset
    train_dataset = raw_dataset['train'].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )

    validation_dataset = raw_dataset['validation'].map(
        preprocessing_validation_example,
        batched=True,
        remove_columns=raw_dataset["validation"].column_names
    )

    train_dataloader, eval_dataloader = process_data_to_model_inputs(train_dataset, validation_dataset)

    # Training
    model, optimizer = model_init()
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    train(train_dataloader, eval_dataloader, model, optimizer, accelerator, validation_dataset, raw_dataset)
    results = fine_tune(model, train_dataset, validation_dataset, raw_dataset)
    print(results)

    # Evaluation
    model_path = os.path.join(MODEL_PATH, 'electra-finetuned-squadv2', 'checkpoint-49410')
    model = ElectraForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    results = evaluate(model, eval_dataloader, validation_dataset, raw_dataset, accelerator)
    print(results)

    # Predict
    model_path = os.path.join(MODEL_PATH, 'electra-finetuned-squadv2', 'checkpoint-49410')
    context = "My name is Sylvain and I live in Paris. " \
              "I am a student at Sorbonne University. " \
              "I am currently working on my master's degree in computer science." \
              "I am also a research engineer at Hugging Face." \
              "I love playing the piano and the guitar." \
              "I have a dog called Pixel."
    question = "Where do I live?"
    prediction = predict(model_path, context, question)
    print(prediction)

if __name__ == '__main__':
    main()
