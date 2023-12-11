# ============================== imorts =======================================
import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title


# ============================== Setup / constants ====================================
# Set up paths
# Set up paths
OS_PATH = os.getcwd()
IMAGE_PATH = os.path.join(OS_PATH, "assets")
os.chdir("../")
os.chdir("../")
ROOT_PATH = os.getcwd()
MODEL_PATH = os.path.join(ROOT_PATH, "Models")
ELECTRA_PATH = os.path.join(MODEL_PATH, "ELECTRA", "electra-finetuned-squadv2")
LORA_BERT_PATH = os.path.join(MODEL_PATH,  "LORA-QA_BERT")
os.chdir(OS_PATH)

# Model Checkpoints
electra_checkpoint = "google/electra-base-discriminator"
electra_finetuned_checkpoint = "checkpoint-49410"
LORA_BERT_checkpoint = "bert-base-uncased"

# ============================== Page ====================================

st.title("Question Answering on SQuAD 2.0")

st.header("Model Description")


# ===================== ELECTRA =====================
st.subheader("ELECTRA")

st.markdown("""
    #### • Two-Component Structure
    #### • Generator: Its role is to replace some tokens in the input data with a plausible alternatives.
    #### • Discriminator: Its role is to determine whether the replaced tokens are real or fake.
    #### • Joint Training: Both the generator and the discriminator are trained simultaneously. (It contrasts with GANs.)
    #### • Final Model Utilization: After training, only the discriminator is used for downstream tasks. (It contrasts with the traditional approach.)
    #### • Efficiency and Scaling: ELECTRA shows that it's more efficient than models like BERT in terms of computational resources needed for training. 
""")

st.image(os.path.join(IMAGE_PATH, "Electra_architecture.png"), use_column_width=True)
st.write("Figure 1: ELECTRA Architecture (Source: Electra Paper)")

# ===================== BERT =====================
st.subheader("BERT with LORA")

st.markdown("""
    #### • Transformer 3, an  attention mechanism :Encoder, Decoder
    #### • Encoder: Its target is to process the input text and understand the context of each word or token within it.
    #### • Decoder: Its target is to take the encoded input and, step by step, produces the output text.
    #### • ATTENTION Mechanism: It gives the model ability to weigh the importance of different words in the sentence relative to each other.
    #### • Embedding Layers: Find the Embedding Representation of Each words (Word Embedding, Positional Embedding, Segment Embedding)
    #### • Pretraining: Marked Language Models/ Next Sentence Prediction
    #### • Fine-Tuning on QA: Start Vector/ End Vector
    #### • Why using LORA? Base BERT model: 144millions parameters 
    #### • A method to learn a lower-dimensional, task-specific representation of the layer’s weights.
""")

st.image(os.path.join(IMAGE_PATH, "BERT_architecture.png"), use_column_width=True)
st.write("Figure 2: BERT Architecture (Source: Google AI Language)")

st.image(os.path.join(IMAGE_PATH, "LORA_architecture.png"), use_column_width=True)
st.write("Figure 3: LORA Architecture ")

# ===================== Training Setups =====================
st.subheader("Training Setups")

st.markdown("""
### ELECTRA Model Fine-Tuning for SQuAD 2.0

#### - **Dataset**: Utilized the SQuAD 2.0 dataset with 129,941 training, 5,951 development, and 5,915 test examples.
#### - **Preprocessing**: Tokenization, sliding windows, padding, and answer localization.
#### - **Evaluation Metrics**: Exact Match (EM) and F1 score, with negative log-likelihood monitoring for overfitting.

#### **Hyperparameters**:
#### - Maximum Sequence Length: 384
#### - Stride: 128
#### - Number of Predictions to Generate: 20
#### - Maximum Answer Length: 30

#### **Training Parameters**:
#### - Learning Rate: 5e-5
#### - Batch Size: 32
#### - Number of Epochs: 3 (Training duration: 1.5 hours)

##### The model is fine-tuned using HuggingFace's implementation with an AdamW optimizer.
""")

st.markdown("""
### BERT Model Fine-Tuning for SQuAD 2.0

#### - **Dataset**: Utilized the SQuAD 2.0 dataset with 129,941 training, 5,951 development, and 5,915 test examples.
#### - **Preprocessing**: The dataset is tokenized using BERT's tokenizer. Each example is processed to identify the start and end positions of answers within the context.
#### - **Evaluation Metrics**: F1 score

#### **Hyperparameters**:
#### - Model: bert-base-uncased with optional LoRA configuration.
#### - Maximum Sequence Length: 512
#### - Learning Rate: 5e-5
#### - Batch Size: 16
#### - Number of Epochs: 3 (Training duration: 2 hours)

#### **LORA Parameters**:
#### - r: 16
#### - lora_alpha=32
#### - lora_dropout=0.05

##### The model is fine-tuned using HuggingFace's implementation with an AdamW optimizer.
""")