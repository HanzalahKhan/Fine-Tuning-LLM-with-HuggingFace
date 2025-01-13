import os
import sys
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, TrainingArguments 
from transformers import Trainer, AutoTokenizer

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, compute_metrics

@dataclass
class ModelTrainerConfig:
    training_info_file_path=os.path.join("artifacts","training_info")
    trained_model_file_path=os.path.join("artifacts","trained_model")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_data,test_data,eval_data):
        try:
            logging.info("Creating the model trainer")

            label_to_id = {x['label_name']:x['label'] for x in train_data}
            id_to_label = {v:k for k,v in label_to_id.items()}

            logging.info("Initializing tokenizer, model and configuration")
            model_ckpt='bert-base-uncased'  
            tokenizer=AutoTokenizer.from_pretrained(model_ckpt)          
            num_labels=len(label_to_id)
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            config=AutoConfig.from_pretrained(model_ckpt, label2id=label_to_id, id2label=id_to_label)
            model=AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)
            
            # Parameters
            batch_size = 64

            training_dir = self.model_trainer_config.training_info_file_path

            # Define TrainingArguments
            training_args = TrainingArguments(
                output_dir=training_dir,           # Directory to save checkpoints and logs
                overwrite_output_dir=True,         # Overwrite output_dir if it already exists
                evaluation_strategy='epoch',       # Evaluate at the end of each epoch
                save_strategy='epoch',             # Save checkpoints at the end of each epoch
                save_total_limit=2,                # Keep only the last 2 checkpoints
                learning_rate=2e-5,                # Learning rate
                per_device_train_batch_size=batch_size,  # Batch size for training
                per_device_eval_batch_size=batch_size,   # Batch size for evaluation
                num_train_epochs=2,                # Number of training epochs
                weight_decay=0.01,                 # Weight decay for regularization
                disable_tqdm=False,                # Show progress bars
                logging_dir='./logs',              # Directory for logs (for TensorBoard/W&B)
                logging_steps=500,                 # Log metrics every 500 steps
                run_name="bert-finetuning-run",    # Experiment name for tracking
                gradient_accumulation_steps=1,     # Adjust if batch size needs to be divided
                load_best_model_at_end=True,       # Automatically load the best checkpoint
                metric_for_best_model="accuracy",  # Metric used to identify the best model
                greater_is_better=True             # Whether a higher value of the metric is better
            )

            logging.info("Initializing the trainer")
            trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_data,
                    eval_dataset=eval_data,
                    compute_metrics=compute_metrics,
                    tokenizer=tokenizer
                )
            
            logging.info("Training is in progress...")
            trainer.train()
            logging.info("Training completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=trainer,
                is_trainer=True
            )
            logging.info("Trained model is saved.")

            logging.info("Prediciting the output on test data")
            pred_output=trainer.predict(test_data)

            return pred_output.metrics
            
        except Exception as e:
            raise CustomException(e,sys)