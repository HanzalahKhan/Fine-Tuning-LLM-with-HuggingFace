import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

id_to_label = {
    1: "joy",
    0: "sadness",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

st.title("Text Classification with Hugging Face Transformers")

classifier=pipeline('sentiment-analysis', model='model', device=-1)

text=st.text_area("Enter the text to be classified")

if st.button("Classify"):
    inputs = classifier.tokenizer(text, return_tensors="pt")  # Tokenize text for the model
    outputs = classifier.model(**inputs)  # Forward pass through the model

    # Move the logits to CPU and convert to numpy
    logits = outputs.logits.cpu().tolist()  # logits contain the raw predictions from the model
    prediction = logits[0].index(max(logits[0]))
    st.write(id_to_label[prediction])

