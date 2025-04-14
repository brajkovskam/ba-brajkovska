"""
data_preparation.py

Loads the MRPC dataset from GLUE, tokenizes it using a BERT tokenizer,
and saves two folders to disk:
  - tokenized_train/
  - tokenized_valid/
"""

import os
import logging
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

TRAIN_DIR = "tokenized_train"
VALID_DIR = "tokenized_valid"

def main():
    logger.info("Loading the MRPC dataset from GLUE...")
    raw_datasets = load_dataset("glue", "mrpc")

    logger.info(f"MRPC dataset splits: {raw_datasets}")

    # Load a BERT tokenizer
    model_name = "bert-base-cased"
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    logger.info("Tokenizing train split...")
    tokenized_train = raw_datasets["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence1", "sentence2", "idx"]
    )

    logger.info("Tokenizing validation split...")
    tokenized_valid = raw_datasets["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence1", "sentence2", "idx"]
    )

    # Save to disk
    for folder_name, ds in [(TRAIN_DIR, tokenized_train), (VALID_DIR, tokenized_valid)]:
        if os.path.exists(folder_name):
            logger.info(f"Removing old '{folder_name}' folder...")
            import shutil
            shutil.rmtree(folder_name)
        logger.info(f"Saving dataset to '{folder_name}'...")
        ds.save_to_disk(folder_name)

    logger.info("Data preparation complete.")

if __name__ == "__main__":
    main()
