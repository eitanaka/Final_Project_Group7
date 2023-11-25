"""
Author: Cody Yu, Pon Swarnalaya Ravichandran, Ei Tanaka
Date: Dec 11, 2023
Purpose: Main file for the project
"""

# ============================= Imports =============================
# ============================= Imports =============================
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from EDA import SQuAD2  # Assuming you have a squad_dataset.py similar to your EDA.py script

# ============================= 1. Loading Dataset =============================
# https://rajpurkar.github.io/SQuAD-explorer/ (SQuAD2.0)
def load_dataset(split):
    dataset = SQuAD2(split=split)
    return dataset

train_dataset = load_dataset('train')
dev_dataset = load_dataset('dev')


# ============================= 2. Data Preprocessing =============================
def preprocess_data(dataset, tokenizer):
    processed_data = []

    for example in dataset:
        # Text normalization: lowercasing, removing punctuation
        context = example['context'].lower()
        question = example['question'].lower()

        # Tokenization and Vectorization
        inputs = tokenizer.encode_plus(question, context,
                                       add_special_tokens=True,
                                       return_tensors='pt',
                                       truncation=True,
                                       padding='max_length',
                                       max_length=512)

        # Handling unanswerable questions
        answerable = 0 if example['is_impossible'] else 1

        processed_data.append({
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'answerable': torch.tensor(answerable)
        })

    return processed_data

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess datasets
preprocessed_train_data = preprocess_data(train_dataset, tokenizer)
preprocessed_dev_data = preprocess_data(dev_dataset, tokenizer)
# ============================= 3. Data Loader =============================
def create_data_loader(processed_data, batch_size=16):
    return DataLoader(processed_data, batch_size=batch_size, shuffle=True)

# Create data loaders
train_data_loader = create_data_loader(preprocessed_train_data)
dev_data_loader = create_data_loader(preprocessed_dev_data)


# ============================= 4. Hyperparameters =============================



# ============================= 5. Optimizing model performance =============================

# ============================= 6. Model =============================

# ============================= 7. Training =============================

# ============================= 8. Testing =============================

# ============================= 9. Visualization =============================

# ============================= Main =============================
