from transformers import pipeline
import numpy

# Load the model on CPU
classifier = pipeline("text-classification", model="model", device=-1)  # -1 for CPU

# Input text
text = "Your input text"

# Get model outputs manually (instead of using the pipeline's default flow)
inputs = classifier.tokenizer(text, return_tensors="pt")  # Tokenize text for the model
outputs = classifier.model(**inputs)  # Forward pass through the model

# Move the logits to CPU and convert to numpy
logits = outputs.logits.cpu().tolist()  # logits contain the raw predictions from the model

print(logits)
