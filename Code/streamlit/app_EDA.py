# ============================== imorts =======================================
import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title
from PIL import Image

# ============================== Setup / constants ====================================
# Set up paths
OS_PATH = os.getcwd()
os.chdir("../../")
ROOT_PATH = os.getcwd()
IMAGE_PATH = os.path.join(OS_PATH, 'assets')
IMAGE_PATH11_t = os.path.join(IMAGE_PATH, "context_len_train.png" )
IMAGE_PATH11_d = os.path.join(IMAGE_PATH, "context_len_dev.png" )
IMAGE_PATH12_t = os.path.join(IMAGE_PATH, "Question_len_train.png")
IMAGE_PATH12_d = os.path.join(IMAGE_PATH, "Question_length_dev.png")
IMAGE_PATH13_t = os.path.join(IMAGE_PATH, "Answer_len_train.png")
IMAGE_PATH13_d = os.path.join(IMAGE_PATH, "Answer_len_dev.png")
IMAGE_PATH14_t = os.path.join(IMAGE_PATH, "ans_vs_unans_train.png")
IMAGE_PATH14_d = os.path.join(IMAGE_PATH, "ans_vs_unans_dev.png" )
IMAGE_PATH15 = os.path.join(IMAGE_PATH, "Most_com_ques_train.jpeg" )
IMAGE_PATH16 = os.path.join(IMAGE_PATH, "most_com_words_ques_dev.jpeg" )
IMAGE_PATH17 = os.path.join(IMAGE_PATH, "distribution dev.jpeg" )
IMAGE_PATH18 = os.path.join(IMAGE_PATH, "distribution dev.jpeg" )
IMAGE_PATH19 = os.path.join(IMAGE_PATH, "Average context question similarity and answer position analysis.jpeg" )
IMAGE_PATH20 = os.path.join(IMAGE_PATH, "comparison.png" )
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

with st.expander("Statiscs about dataset"):
    st.write("""
            <div style="font-size: 18px;">
                <p>Number of training examples: 130319</p>
                <p>Number of dev examples: 11873</p>
            </div>
        """, unsafe_allow_html=True)

# List of options for the radio button
options = ['Context Length Analysis Train',
           'Context Length Analysis Test',
           'Question Length Analysis Train',
           'Question Length Analysis Test',
           'Answer Length Analysis Train',
           'Answer Length Analysis Test',
           'Answerable vs Unanswerable Questions Train',
           'Answerable vs Unanswerable Questions Test',
           'Word Length Analysis',
           'Distribution of question type train',
           'Distribution of question type dev'
           'Answer context similarity and Answer position analysis']

# Create a radio button in Streamlit
selected_option = st.radio("Select an option:", options)

# Dictionary mapping options to corresponding image paths
image_paths = {
'Context Length Analysis Train': IMAGE_PATH11_t,
'Context Length Analysis Test': IMAGE_PATH11_d,
'Question Length Analysis Train': IMAGE_PATH12_t,
'Question Length Analysis Test': IMAGE_PATH12_d,
'Answer Length Analysis Train':IMAGE_PATH13_t,
'Answer Length Analysis Test':IMAGE_PATH13_d,
'Answerable vs Unanswerable Questions Train': IMAGE_PATH14_t,
'Answerable vs Unanswerable Questions Test': IMAGE_PATH14_d,
'Word Length Analysis':IMAGE_PATH15,
'Distribution of question type train': IMAGE_PATH16,
'Distribution of question type test': IMAGE_PATH17,
'Answer context similarity and Answer position analysis': IMAGE_PATH19,
}

# Display the selected image
if selected_option in image_paths:
    selected_image_path = image_paths[selected_option]
    st.image(selected_image_path, caption=f"Selected Option: {selected_option}", use_column_width=True)
else:
    st.warning("Please select an option to view the corresponding image.")

st.write("The below figure is the comparison of the SQuAD 1.1 and SQuAD 2.0 representing the train, development and test values "
         "of the total examples, negative examples, total articles and articles with negatives ")

img2 = Image.open(IMAGE_PATH20)
st.image(img2)
