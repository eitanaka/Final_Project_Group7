# ============================== imorts =======================================
import os
import streamlit as st
import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import ElectraTokenizer, ElectraForQuestionAnswering, BertTokenizerFast, BertForQuestionAnswering
# ============================== Setup / constants ====================================
# Set up paths
OS_PATH = os.getcwd()
os.chdir("../../")
ROOT_PATH = os.getcwd()
MODEL_PATH = os.path.join(ROOT_PATH, "Models")
os.chdir(OS_PATH)

# Model Checkpoints
electra_finetuned_checkpoint = "checkpoint-49410"
electra_check_point = "google/electra-base-discriminator"
BERT = "qa-bert-base"
LORA_BERT_checkpoint = "LORA-QA_BERT"

# ============================== function ====================================
def predict(model_name, question, context):
    # Load the model
    if model_name == "ELECTRA":
        model_path = os.path.join(MODEL_PATH, "ELECTRA", "electra-finetuned-squadv2", electra_finetuned_checkpoint)
        model = ElectraForQuestionAnswering.from_pretrained(model_path)
        tokenizer = ElectraTokenizer.from_pretrained(electra_check_point)

    elif model_name == "BERT":
        model_path = os.path.join(MODEL_PATH, BERT)
        model = BertForQuestionAnswering.from_pretrained(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(model_path)

    elif model_name == "BERT_with_LORA":
        model_path = os.path.join(MODEL_PATH, LORA_BERT_checkpoint)
        config = PeftConfig.from_pretrained(model_path)
        model = PeftModel.from_pretrained(BertForQuestionAnswering.from_pretrained(config.base_model_name_or_path),
                                          model_path)
        tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name_or_path)

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs).to_tuple()

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# ============================== Page ====================================
st.title("Question Answering on SQuAD 2.0")

st.header("Demo: Free Question Answering")

st.sidebar.title("Model Selection: ")
model_name = st.sidebar.selectbox("Select Model", ("BERT", "ELECTRA", "BERT_with_LORA"))

# Initialize session state for question and context
if 'context' not in st.session_state:
    st.session_state['context'] = ""
if 'question' not in st.session_state:
    st.session_state['question'] = ""

# Function to update session state from input
def update_context():
    st.session_state['context'] = st.session_state.new_context

def update_question():
    st.session_state['question'] = st.session_state.new_question

# Function to clear inputs
def clear_inputs():
    st.session_state['context'] = ""
    st.session_state['question'] = ""

# Get the question and context with callback to update session state
context = st.text_area("Context:", value=st.session_state['context'], key='new_context', on_change=update_context)
question = st.text_input("Question:", value=st.session_state['question'], key='new_question', on_change=update_question)

# Layout for buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            answer = predict(model_name, question, context)
            st.write(answer.capitalize())

with col2:
    if st.button("Clear", on_click=clear_inputs):
        pass
