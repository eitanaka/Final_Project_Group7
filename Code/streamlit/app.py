"""
Author: Ei Tanaka, Cody Yu, Pon Swarnalaya Ravichandran
Date: December 11, 2023
Purpose: This file is used to run the streamlit app
References:
    https://github.com/blackary/st_pages?tab=readme-ov-file (streamlit pages)
"""

# ============================== imorts =======================================
import os
from tkinter import Image
from PIL import Image
import streamlit as st
from st_pages import Page, show_pages, add_page_title


# ============================== Setup / constants ====================================
# Set up paths
OS_PATH = os.getcwd()
IMAGE_PATH = os.path.join(OS_PATH, 'assets')
os.chdir("../../")
ROOT_PATH = os.getcwd()
IMAGE_PATH1 = os.path.join(IMAGE_PATH, "FIGURE01.png")
IMAGE_PATH2 = os.path.join(IMAGE_PATH, "Figure_nlp_task.png")
IMAGE_PATH3 = os.path.join(IMAGE_PATH, "Fig_03.png")
IMAGE_PATH4 = os.path.join(IMAGE_PATH, "Fig 04.png")
MODEL_PATH = os.path.join(ROOT_PATH, "Models")
ELECTRA_PATH = os.path.join(MODEL_PATH, "ELECTRA", "electra-finetuned-squadv2")
XLNET_PATH = ""
os.chdir(OS_PATH)

# Model Checkpoints
electra_checkpoint = "google/electra-base-discriminator"
electra_finetuned_checkpoint = "checkpoint-49410"
XLNET_checkpoint = ""
XLNET_finetuned_checkpoint = ""

# Set up pages
st.set_page_config(layout="wide")

# Specify what pages should be shown in the sidebar, and what their labels should be
show_pages(
    [
        Page("app.py", "Introduction"),
        Page("app_EDA.py", "Exploratory Data Analysis"),
        Page("app_Model.py", "Model"),
        Page("app_Results.py", "Results"),
        Page("app_demo.py", "Demo"),
    ]
)

# =============.================= Main Contents ====================================
st.title("Question Answering on SQuAD 2.0")

st.header("Introduction")

st.write("In the dynamic realm of Natural Language Processing (NLP), the advent of question-answering (QA) systems marks a significant stride in our ability to interact with and process digital information. "
         "This project is dedicated to developing a reading comprehension-based QA system inspired by the comprehensive insights in Speech and Language Processing by Daniel Jurafsky & James H. Martin (2023). "
         "We aim to create a system that can interpret and respond to questions posed in natural language, drawing answers from provided text passages.")

st.header("NLP Task - Question Answering")

st.write("In the Natural Language Processing (NLP) field, extractive Question Answering (QA) is a pivotal task involving locating the answer to a question within a specified text passage. "
         "This task is inherently challenging, as it requires the system to comprehend the posed question and accurately extract the specific text portion that contains the answer. "
         "As detailed in Hugging Face's documentation and task library (n.d.), "
         "extractive QA demands the capability to sift through extensive text and pinpoint information that precisely responds to the query.")

st.write("Figure 01")
img2 = Image.open(IMAGE_PATH2)
st.image(img2)

st.write("Jurafsky and Martin (2023), in their seminal work," "Speech and Language Processing," "elucidate the complexities of extractive QA, highlighting the necessity for advanced NLP techniques and models."
"These models are crucial for understanding the context and semantics embedded in both the question and the passage, thus enabling the identification of the exact text span that answers the question."
"Extractive QA is particularly vital in scenarios necessitating factual answers directly sourced from the provided text, such as in academic research or specific information retrieval tasks."
"In our project, we embrace the challenges of extractive QA by training our model on the SQuAD 2.0 dataset. "
"This dataset, encompassing diverse questions and passages, provides a comprehensive framework for the system to learn from varying contexts and question types." ""
"The model is meticulously trained to parse the subtleties of language in questions and passages, enhancing its ability to discern and extract the relevant answers accurately." 
"This endeavor underscores the significance of sophisticated text processing and comprehension in NLP, laying the groundwork for more intelligent and adept information retrieval systems.")

st.header("Dataset")

st.markdown("""
#### - SQuAD 2.0: Stanford Question Answering Dataset 2.0 (SQuAD 2.0) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles.
#### - Introduction of Unanswerable Questions: SQuAD 2.0 is enhanced with 53,775 unanswerable questions, created by crowdworkers, in addition to the answerable ones from SQuAD 1.1.
#### - Objective of SQuAD 2.0: It aims to challenge machine learning models in determining when a correct answer is not present in the text, elevating their proficiency in reading comprehension.
#### - Increased Complexity: Unlike SQuAD 1.1, which focused solely on finding the correct text span for a given question, SQuAD 2.0 adds the dimension of identifying unanswerable questions, making it more complex.
#### - Public Benchmark: SQuAD 2.0 is available publicly and serves as the primary benchmark on the official SQuAD leaderboard.
""")

st.write("Figure 02")
img = Image.open(IMAGE_PATH1)
st.image(img)

st.write("Figure 03")
img3 = Image.open(IMAGE_PATH3)
st.image(img3)

st.write("Figure 04")
img4 = Image.open(IMAGE_PATH4)
st.image(img4)

