# ==================================== import/setup ====================================
import os
import json
import matplotlib.pyplot as plt
import requests
from torch.utils.data import Dataset

# ==================================== Load Dataset ====================================
# Load data
# The dataset is from "The Standard Question Answering Dataset" (SQuAD 2.0)
# https://rajpurkar.github.io/SQuAD-explorer/
class SQuAD2(Dataset):
    def __init__(self, split='train'):
        self.data = []

        # URLs for train and dev sets
        urls = {
            "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
            "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        }

        # Ensure the split is valid
        if split not in urls:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'dev'.")

        # Download and parse the dataset
        self.download_and_parse(urls[split])

    def download_and_parse(self, url):
        response = requests.get(url)
        data = response.json()

        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']

                for qa in paragraph['qas']:
                    question = qa['question']
                    id = qa['id']
                    answers = [answer['text'] for answer in qa['answers']]

                    # Handle the possibility of unanswerable questions in SQuAD 2.0
                    is_impossible = qa.get('is_impossible', False)

                    self.data.append({
                        'id': id,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'is_impossible': is_impossible
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx, context, question, answers, is_impossible
        return self.data[idx]

# ==================================== Helper Function for EDA ====================================
# Length Analysis
def get_length(dataset, field):
    """
    Get the length of each example in a dataset for a given field.
    :param dataset:
    :param field:
    :return:
    """
    context_lengths = []

    for example in dataset:
        context_lengths.append(len(example[field]))

    return context_lengths

def plot_length_distribution(data, title, bins=50):
    """
    Plot the length distribution of a dataset field.

    Parameters:
    data (list): List of lengths for the dataset field.
    title (str): Title of the plot.
    bins (int): Number of bins in the histogram.
    """
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=bins, range=(0, max(data)))  # Set the range based on the max value in data
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

# Answer Length Analysis
def get_answer_length(dataset):
    """
    Get the length of each answer in a dataset.
    :param dataset:
    :return:
    """
    answer_lengths = []

    for example in dataset:
        for answer in example['answers']:
            answer_lengths.append(len(answer))

    return answer_lengths

# Answerable vs Unanswerable Questions
def get_answerable(dataset):
    """
    Get the number of answerable and unanswerable questions in a dataset.
    :param dataset:
    :return:
    """
    answerable = 0
    unanswerable = 0

    for example in dataset:
        if example['is_impossible']:
            unanswerable += 1
        else:
            answerable += 1

    return answerable, unanswerable

def plot_answerable(answerable, unanswerable, title):
    plt.figure(figsize=(12, 8))
    plt.bar(['Answerable', 'Unanswerable'], [answerable, unanswerable])
    plt.title(title)
    plt.xlabel('Answerable')
    plt.ylabel('Frequency')
    plt.show()

# Word Frequency Analysis

# Distribution of Question Types

# Context-Question Similarity

# Answer Position Analysis

# Visualizations

# Named Entity Analysis

def main():
    # Load the dataset
    train_dataset = SQuAD2(split='train')
    dev_dataset = SQuAD2(split='dev')

    # EDA
    # Print some stats about the dataset
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of dev examples: {len(dev_dataset)}")

    # Print the datatype of an example
    print('--- Datatype of an example ---')
    print(type(train_dataset[0]))

    # Print an example from the training set (first 3)
    print('--- Example from training set ---')
    print('ID: ', train_dataset[0]['id'])
    print('Context: ', train_dataset[0]['context'])
    print('Question: ', train_dataset[0]['question'])
    print('Answer: ', train_dataset[0]['answers'])
    print('Is Impossible: ', train_dataset[0]['is_impossible'])

    # Print an example from the dev set (first 3)
    print('--- Example from dev set ---')
    print('ID: ', dev_dataset[0]['id'])
    print('Context: ', dev_dataset[0]['context'])
    print('Question: ', dev_dataset[0]['question'])
    print('Answer: ', dev_dataset[0]['answers'])
    print('Is Impossible: ', dev_dataset[0]['is_impossible'])

    # Context Length Analysis
    train_context_lengths = get_length(train_dataset, 'context')
    dev_context_lengths = get_length(dev_dataset, 'context')
    plot_length_distribution(train_context_lengths, 'Train Context Lengths')
    plot_length_distribution(dev_context_lengths, 'Dev Context Lengths')

    # Question Length Analysis
    train_question_lengths = get_length(train_dataset, 'question')
    dev_question_lengths = get_length(dev_dataset, 'question')
    plot_length_distribution(train_question_lengths, 'Train Question Lengths', bins=50)
    plot_length_distribution(dev_question_lengths, 'Dev Question Lengths', bins=50)

    # Answer Length Analysis
    train_answer_lengths = get_answer_length(train_dataset)
    dev_answer_lengths = get_answer_length(dev_dataset)
    plot_length_distribution(train_answer_lengths, 'Train Answer Lengths', bins=50)
    plot_length_distribution(dev_answer_lengths, 'Dev Answer Lengths', bins=50)

    # Answerable vs Unanswerable Questions
    train_answerable, train_unanswerable = get_answerable(train_dataset)
    dev_answerable, dev_unanswerable = get_answerable(dev_dataset)
    plot_answerable(train_answerable, train_unanswerable, 'Train Answerable vs Unanswerable')
    plot_answerable(dev_answerable, dev_unanswerable, 'Dev Answerable vs Unanswerable')


if __name__ == '__main__':
    main()



