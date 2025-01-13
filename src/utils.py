import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.metrics import accuracy_score, f1_score
from src.exception import CustomException

import pickle
import os


def save_object(file_path, obj, is_trainer=False):
    try:
        dir_path = os.path.dirname(file_path)

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Save Hugging Face Trainer model
        if is_trainer and hasattr(obj, "save_model"):
            obj.save_model(dir_path)  # Save model directly to the directory
        else:
            # Save other objects (e.g., regular Python objects) using pickle
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average='weighted')
  acc = accuracy_score(labels, preds)
  return {'accuracy': acc, 'f1': f1}
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)