# ====================================== Import ======================================
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, Dataset
import transformers
from transformers import BertTokenizerFast, BertForQuestionAnswering
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from tqdm import tqdm
import timeit
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import random
import numpy as np


# ====================================== Setup ======================================
OS_PATH = os.getcwd()
os.chdir("..")
ROOT_PATH = os.getcwd()
MODEL_PATH = os.path.join(ROOT_PATH, "Models", "BERT")
os.chdir(OS_PATH)

model_name = 'BERT-finetuned-squadv2'

# Random seed for reproducibility
manual_seed = 42
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)  # For CUDA if available

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")



# ====================================== Hyperparameters ======================================
# Dataset paths
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, "Code", "squad", "train-v2.0.json")
DEV_DATA_PATH = os.path.join(ROOT_PATH, "Code", "squad", "dev-v2.0.json")

# Model configuration
MODEL_PATH = "bert-base-uncased"
LORA = True
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
EPOCHS = 3

# Path for saving the trained model
MODEL_SAVE_PATH = os.path.join(ROOT_PATH, "Code", "models", f"{MODEL_PATH}-lr{LEARNING_RATE}-epochs{EPOCHS}-batchsize{BATCH_SIZE}-LORA-retrain")

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

# ====================================== Function ======================================
class SquadDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        contexts, questions, answers = self.read_data(data_path)
        answers = self.add_end_idx(contexts, answers)
        #
        encodings = tokenizer(contexts, questions, padding=True, truncation=True)
        self.encodings = self.update_start_end_positions(encodings, answers, tokenizer)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    def read_data(self, path):
        with open(path, 'rb') as f:
            squad = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in squad['data']:
            for parag in group['paragraphs']:
                context = parag['context']
                for qa in parag['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        return contexts, questions, answers

    def add_end_idx(self, contexts, answers):
        '''calculates the end index of each answer in the context based '''
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]
            start_idx = answer["answer_start"]

            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer["answer_start"] = start_idx - 1
                answer["answer_end"] = end_idx - 1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer["answer_start"] = start_idx - 2
                answer["answer_end"] = end_idx - 2
        return answers

    def update_start_end_positions(self, encodings, answers, tokenizer):
        '''updates the tokenized data with the start and end position of answer
        These positions are transformed from character-level indices in the original text to
         token-level indices in the tokeninzed text'''
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"]-1))
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings["start_positions"] = start_positions
        encodings["end_positions"] = end_positions

        return encodings

# ====================================== Model ======================================
def model_init():
        # Initialize the model
    model = BertForQuestionAnswering.from_pretrained(MODEL_PATH).to(device)

        # Apply LoRA configuration if enabled
    if LORA:
        lora_config = LoraConfig(
                task_type="QUESTION_ANS",
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                fan_in_fan_out=False,
                bias="none",
            )
            # Adding LoRA to the model
        model = get_peft_model(model, lora_config)
        print("# Trainable Parameters After LoRA")
        model.print_trainable_parameters()

        # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer


# ====================================== Training ======================================
def train(train_dataloader, val_dataloader, model, optimizer):
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        train_running_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        # Calculate average loss over the epoch
        train_loss = train_running_loss / len(train_dataloader)

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                val_running_loss += outputs.loss.item()

        # Calculate average validation loss
        val_loss = val_running_loss / len(val_dataloader)

        # Print training and validation loss
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
# ====================================== Compute Metrics ======================================
def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)

def compute_f1(a_gold, a_pred):
    gold_toks = a_gold.split()
    pred_toks = a_pred.split()
    common = set(gold_toks) & set(pred_toks)
    num_same = len(common)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(predictions, true_answers):
    exact_matches = []
    f1_scores = []
    for pred, true in zip(predictions, true_answers):
        exact_match = compute_exact(true, pred)
        f1_score = compute_f1(true, pred)
        exact_matches.append(exact_match)
        f1_scores.append(f1_score)
    avg_exact_match = sum(exact_matches) / len(exact_matches)
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    return {
        "average_exact_match": avg_exact_match,
        "average_f1_score": avg_f1_score
    }


# ====================================== Main ======================================
def main():
    # Load and preprocess the dataset
    train_dataset = SquadDataset('/home/ubuntu/NLP/NLP_Final_Project_Group7/Code/squad/train-v2.0.json', tokenizer)
    dev_dataset = SquadDataset('/home/ubuntu/NLP/NLP_Final_Project_Group7/Code/squad/dev-v2.0.json', tokenizer)

    # Create DataLoaders for our training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    # Initialize model and optimizer
    model, optimizer = model_init()

    # Train the model
    train(train_dataloader, val_dataloader, model, optimizer)

    # Evaluate the model
    # You'll need to implement your own function to get predictions from your model
    predictions = get_predictions(model, val_dataloader)
    true_answers = [item['answers'] for item in dev_dataset]

    # Compute metrics
    metrics = compute_metrics(predictions, true_answers)
    print("Evaluation Metrics:", metrics)

# Utility function to get predictions from the model
def get_predictions(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            # Extract and process the model's output here
            # Add the processed output to the predictions list
            # You'll need to modify this part based on your model's specific output format
            predictions.append(outputs)
    return predictions

if __name__ == '__main__':
    main()
