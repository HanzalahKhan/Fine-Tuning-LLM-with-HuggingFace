import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    tokenized_obj_file_path=os.path.join('artifacts',"tokenized_data")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path,eval_path):

        try:
            logging.info("Entered the data transformation method or component")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            eval_df=pd.read_csv(eval_path)
            train_df['words_per_tweet']=train_df['text'].str.split().apply(len)
            test_df['words_per_tweet']=test_df['text'].str.split().apply(len)
            eval_df['words_per_tweet']=eval_df['text'].str.split().apply(len)
            logging.info("Reading train, test and val data is completed")

            logging.info("Tokenizing the text data")
            dataset = DatasetDict({
                            'train': Dataset.from_pandas(train_df, preserve_index=False),
                            'test': Dataset.from_pandas(test_df, preserve_index=False),
                            'eval': Dataset.from_pandas(eval_df, preserve_index=False)
                        })
            
            model_ckpt='bert-base-uncased'
            tokenizer=AutoTokenizer.from_pretrained(model_ckpt)

            def tokenize(batch):
                return tokenizer(batch['text'], padding=True, truncation=True)

            emotion_embeded=dataset.map(tokenize, batched=True, batch_size=None)
            logging.info("Tokenization of the text data is completed")

            save_object(self.data_transformation_config.tokenized_obj_file_path, emotion_embeded)
            logging.info("Tokenized data is saved as pickle file")

            return (
                emotion_embeded['train'],
                emotion_embeded['test'],
                emotion_embeded['eval'],
                self.data_transformation_config.tokenized_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        