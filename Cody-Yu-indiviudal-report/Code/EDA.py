# ==================================== import/setup ====================================
import os
import json
import torch
import matplotlib.pyplot as plt
import requests
from torch.utils.data import Dataset
import re
from collections import Counter
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)
#
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
def word_frequency_analysis(dataset, field):
    """
    Analyze word frequency in a specific field of the dataset.

    :param dataset: The dataset containing the text data.
    :param field: The field of the dataset to analyze (e.g., 'context', 'question').
    :return: A Counter object with word frequencies.
    """
    word_freq = Counter()

    for example in dataset:
        text = example[field].lower()  # Convert to lowercase to count all variations of a word the same
        words = re.findall(r'\w+', text)  # Extract words using regex
        word_freq.update(words)

    return word_freq

def plot_word_frequencies(word_freq, title, num_words=10):
    """
    Plot the most common words and their frequencies.

    Parameters:
    word_freq (Counter): Counter object with word frequencies.
    title (str): Title of the plot.
    num_words (int): Number of most common words to display.
    """
    # Extract the most common words and their frequencies
    common_words = word_freq.most_common(num_words)
    words, frequencies = zip(*common_words)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.bar(words, frequencies)
    plt.title(title)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.show()

# Distribution of Question Types
def question_type_distribution(dataset):
    """
    Analyze the distribution of question types based on their starting words.

    :param dataset: The dataset containing the questions.
    :return: A Counter object with the frequency of each question type.
    """
    question_types = Counter()
    for example in dataset:
        question_words = example['question'].strip().split()
        if question_words:
            # Consider the first word as the question type
            question_type = question_words[0].lower()
            question_types[question_type] += 1
    return question_types

def plot_question_types(question_types, title, num_types=5):
    """
    Plot the distribution of the top question types.

    Parameters:
    question_types (Counter): Counter object with the frequency of each question type.
    title (str): Title of the plot.
    num_types (int): Number of top types to display.
    """
    # Get the top question types
    top_types = question_types.most_common(num_types)
    types, frequencies = zip(*top_types)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.bar(types, frequencies, color='skyblue')
    plt.title(title)
    plt.xlabel('Question Types')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()



# Context-Question Similarity



def get_embeddings(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def context_question_similarity_gpu(dataset):
    similarities = []

    for example in dataset:
        context_embedding = get_embeddings(example['context'])
        question_embedding = get_embeddings(example['question'])

        similarity = cosine_similarity([context_embedding], [question_embedding])[0][0]
        similarities.append(similarity)

    return similarities

# Answer Position Analysis
def answer_position_analysis(dataset):
    """
    Analyze the positions of answers within the contexts in the dataset.

    :param dataset: The dataset containing the context and answers fields.
    :return: A list of answer positions (as a percentage of context length).
    """
    positions = []

    for example in dataset:
        context = example['context']
        context_length = len(context)

        for answer in example['answers']:
            answer_start = context.find(answer)
            if answer_start != -1:  # Check if the answer is found in the context
                # Calculate the position as a percentage of the context length
                position_percentage = (answer_start / context_length) * 100
                positions.append(position_percentage)

    return positions

# Visualizations

# Named Entity Analysis

def named_entity_analysis(dataset, field):
    """
    Perform Named Entity Recognition on a specified field of the dataset.

    :param dataset: The dataset containing the text data.
    :param field: The field of the dataset to analyze (e.g., 'context', 'question').
    :return: A Counter object with the frequency of each named entity label.
    """
    nlp = spacy.load("en_core_web_md")
    entity_labels = Counter()

    for example in dataset:
        doc = nlp(example[field])
        for ent in doc.ents:
            entity_labels[ent.label_] += 1

    return entity_labels

def plot_named_entity_frequencies(entity_labels, title, num_entities=10):
    """
    Plot the most common named entity labels and their frequencies.

    Parameters:
    entity_labels (Counter): Counter object with named entity label frequencies.
    title (str): Title of the plot.
    num_entities (int): Number of most common entities to display.
    """
    # Extract the most common entity labels and their frequencies
    common_entities = entity_labels.most_common(num_entities)
    labels, frequencies = zip(*common_entities)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.bar(labels, frequencies, color='skyblue')
    plt.title(title)
    plt.xlabel('Entity Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.show()


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

    # World Frequency Analysis
    train_context_word_freq = word_frequency_analysis(train_dataset, 'context')
    plot_word_frequencies(train_context_word_freq, 'Most Common Words in Training Context', num_words=10)

    train_question_word_freq = word_frequency_analysis(train_dataset, 'question')
    plot_word_frequencies(train_question_word_freq, 'Most Common Words in Training Questions', num_words=10)

    dev_context_word_freq = word_frequency_analysis(dev_dataset, 'context')
    plot_word_frequencies(dev_context_word_freq, 'Most Common Words in Dev Context', num_words=10)

    dev_question_word_freq = word_frequency_analysis(dev_dataset, 'question')
    plot_word_frequencies(dev_question_word_freq, 'Most Common Words in Dev Questions', num_words=10)

    # Distribution of Question Types
    print('--- Distribution of Question Types ---')
    train_question_types = question_type_distribution(train_dataset)
    print("Top 5 Question Types in Training Dataset:", train_question_types.most_common(5))
    plot_question_types(train_question_types, 'Top 5 Question Type Distribution in Training Dataset')

    dev_question_types = question_type_distribution(dev_dataset)
    print("Top 5 Question Types in Dev Dataset:", dev_question_types.most_common(5))
    plot_question_types(dev_question_types, 'Top 5 Question Type Distribution in Dev Dataset')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)
    # Context-Question Similarity Analysis
    train_similarities = context_question_similarity_gpu(train_dataset)
    print("Average Context-Question Similarity in Training Dataset (GPU):",
          sum(train_similarities) / len(train_similarities))
    dev_similarities = context_question_similarity_gpu(dev_dataset)
    print("Average Context-Question Similarity in dev_dataset (GPU):", sum(dev_similarities) / len(dev_similarities))

    # Answer Position Analysis
    train_positions = answer_position_analysis(train_dataset)
    print("Average Answer Position in Training Dataset:", sum(train_positions) / len(train_positions))
    dev_positions = answer_position_analysis(dev_dataset)
    print("Average Answer Position in Dev Dataset:", sum(dev_positions) / len(dev_positions))

    # Named Entity Analysis and Plotting
    train_entity_labels = named_entity_analysis(train_dataset, 'context')
    plot_named_entity_frequencies(train_entity_labels, 'Named Entity Labels in Training Dataset Context',
                                  num_entities=10)

    dev_entity_labels = named_entity_analysis(dev_dataset, 'context')
    plot_named_entity_frequencies(dev_entity_labels, 'Named Entity Labels in Dev Dataset Context', num_entities=10)


if __name__ == '__main__':
    main()



