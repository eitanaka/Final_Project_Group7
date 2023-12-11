# ============================== imorts =======================================
import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title

# ============================== Setup / constants ====================================
# Set up paths
OS_PATH = os.getcwd()
os.chdir("../../")
ROOT_PATH = os.getcwd()
IMAGE_PATH = os.path.join(OS_PATH, 'assets')
MODEL_PATH = os.path.join(ROOT_PATH, "Models")
ELECTRA_PATH = os.path.join(MODEL_PATH, "ELECTRA", "electra-finetuned-squadv2")
XLNET_PATH = ""
os.chdir(OS_PATH)

# Model Checkpoints
electra_checkpoint = "google/electra-base-discriminator"
electra_finetuned_checkpoint = "checkpoint-49410"
XLNET_checkpoint = ""
XLNET_finetuned_checkpoint = ""

# ============================== Page ====================================
st.title("Question Answering on SQuAD 2.0")

st.header("Exploratory Data Analysis (EDA)")

# Add contents for the page here.
genre = st.radio(
        "What you wanna explore",
        ["*Frequency of answers*", "Length analysis", "Answerable vs Unanswerable"],
        captions = ["Know it", "measure it", "question it"],
        index=None,
    )

st.write("You selected:", genre)