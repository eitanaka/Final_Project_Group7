# ============================== imorts =======================================
import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title

# ============================== Setup / constants ====================================
# Set up paths
OS_PATH = os.getcwd()
MODEL_PATH = os.path.join(OS_PATH, "Models")
os.chdir("../../")
ROOT_PATH = os.getcwd()
os.chdir(OS_PATH)

# Model Checkpoints
electra_checkpoint = "google/electra-base-discriminator"
electra_finetuned_checkpoint = "checkpoint-49410"
LORA_BERT_checkpoint = "bert-base-uncased"

# ============================== Page ====================================
st.title("Question Answering on SQuAD 2.0")

st.header("Evaluation / Results")

st.subheader("ELECTRA")
st.markdown("""
    #### • F1 Score: 0.50071
    #### • Exact Match: 0.50071
""")

st.subheader("BERT with LORA")
st.markdown("""
    #### • All params : 109,484,548
    #### • Trainable Parameters: 54% 591,362
    #### • Training Loss(epoch 1 -3): 2.6399 1.6731 1.4301
    #### • Valid Loss(epoch 1 -3): 1.7555 1.3669 1.2309
    #### • F1 Score: 0.50989
""")

# Add contents for the page here.