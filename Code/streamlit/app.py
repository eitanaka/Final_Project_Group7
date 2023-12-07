"""
Author: Ei Tanaka, Cody Yu, Pon Swarnalaya Ravichandran
Date: December 11, 2023
Purpose: This file is used to run the streamlit app
References:
    https://github.com/blackary/st_pages?tab=readme-ov-file (streamlit pages)
"""

# ============================== imorts =======================================
import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title

# ============================== Setup / constants ====================================
# Set up paths
OS_PATH = os.getcwd()
os.chdir("../../")
ROOT_PATH = os.getcwd()
MODEL_PATH = os.path.join(ROOT_PATH, "Models")
ELECTRA_PATH = os.path.join(MODEL_PATH, "ELECTRA", "electra-finetuned-squadv2")
XLNET_PATH = ""
os.chdir(OS_PATH)

# Model Checkpoints
electra_checkpoint = "google/electra-base-discriminator"
electra_finetuned_checkpoint = "checkpoint-49410"
XLNET_checkpoint = ""
XLNET_finetuned_checkpoint = ""

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

# Add contents for the page here.

