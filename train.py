#!/usr/bin/env python

"""
train.py

A complete PyTorch script to fine-tune a BERT model on MRPC using MPS if available.
It saves the final model in the folder 'final_model' along with the tokenizer.
If 'safetensors' is installed, the weights will be saved as 'model.safetensors';
otherwise, 'pytorch_model.bin' is used.
"""

import time
import logging
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer
)
from datasets import load_from_disk

# For advanced metrics
import evaluate
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --------------------
# Hyperparameters
# --------------------
LEARNING_RATE = 3e-5
BATCH_SIZE = 24
NUM_EPOCHS = 6

# Choose device: MPS if available (Apple Silicon), else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --------------------
# Data Loader Function
# --------------------
def create_dataloader(dataset, batch_size, shuffle=True):
    """
    Converts a Hugging Face dataset to a PyTorch DataLoader.
    Expects 'input_ids', 'attention_mask', 'label'.
    """
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# --------------------
# Main Training Function
# --------------------
def main():
    # 1. Load tokenized MRPC data
    logger.info("Loading tokenized train/valid sets from disk...")
    train_dataset = load_from_disk("tokenized_train")
    valid_dataset = load_from_disk("tokenized_valid")
    logger.info(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    # 2. Create DataLoaders
    train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, BATCH_SIZE, shuffle=False)

    # 3. Load model + config (BERT base, 2 labels for MRPC) and tokenizer
    model_name = "bert-base-cased"
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # 4. Create optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Time to First Training Step (TTFT) measurement
    logger.info("Measuring Time to First Training Step (TTFT)...")
    start_time = time.time()

    # Get the first batch from train_loader
    first_batch = next(iter(train_loader))
    for k, v in first_batch.items():
        first_batch[k] = v.to(device)

    model.train()
    outputs = model(
        input_ids=first_batch["input_ids"],
        attention_mask=first_batch["attention_mask"]
    )
    logits = outputs.logits
    loss = F.cross_entropy(logits, first_batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ttft = time.time() - start_time
    logger.info(f"TTFT finished: {ttft * 1000:.2f} ms, initial loss={loss.item():.4f}")

    # 6. Training loop
    total_tokens = 0
    total_time_spent = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        model.train()
        epoch_start = time.time()
        batch_losses = []
        batch_count = 0

        for batch in train_loader:
            batch_count += 1

            # Move to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Count tokens
            tokens_in_batch = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            total_tokens += tokens_in_batch

            # Forward + backward
            step_start = time.time()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            loss = F.cross_entropy(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            total_time_spent += step_time
            batch_losses.append(loss.item())

            # Simple console log
            logger.info(
                f"  [Epoch {epoch}, Batch {batch_count}] "
                f"loss={loss.item():.4f}, step_time={step_time:.2f}s, tokens={tokens_in_batch}"
            )

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(batch_losses) if batch_losses else float('nan')
        tps = total_tokens / total_time_spent if total_time_spent > 0 else float('inf')

        logger.info(
            f"Epoch {epoch} done in {epoch_time:.2f}s. "
            f"Avg Loss={avg_loss:.4f}, total_tokens={total_tokens}, "
            f"time_spent={total_time_spent:.2f}s, TPS={tps:.2f}"
        )

        # 7. Validation
        model.eval()
        correct = 0
        total = 0
        preds_list = []
        labels_list = []

        val_start = time.time()
        with torch.no_grad():
            for batch in valid_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                # Simple accuracy
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])

                # For advanced metrics
                preds_list.extend(preds.cpu().tolist())
                labels_list.extend(batch["labels"].cpu().tolist())

        val_time = time.time() - val_start
        accuracy = correct / total if total else 0.0

        # ---------- Evaluate Library ----------
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        f1_val = f1_metric.compute(predictions=preds_list, references=labels_list)["f1"]
        precision_val = precision_metric.compute(predictions=preds_list, references=labels_list)["precision"]
        recall_val = recall_metric.compute(predictions=preds_list, references=labels_list)["recall"]

        # ---------- Scikit-learn ----------
        f1_sklearn = f1_score(labels_list, preds_list)
        precision_sklearn = precision_score(labels_list, preds_list)
        recall_sklearn = recall_score(labels_list, preds_list)

        logger.info(
            f"Validation accuracy={accuracy*100:.2f}%, samples={total}, time={val_time:.2f}s\n"
            f"HuggingFace Evaluate => F1={f1_val:.3f}, Precision={precision_val:.3f}, Recall={recall_val:.3f}\n"
            f"scikit-learn => F1={f1_sklearn:.3f}, Precision={precision_sklearn:.3f}, Recall={recall_sklearn:.3f}"
        )

    # 8. Save the final model and tokenizer
    logger.info("Saving model and tokenizer to 'final_model'...")
    tokenizer.save_pretrained("final_model")
    model.save_pretrained("final_model")
    logger.info("Training complete. Everything saved to 'final_model'.")

if __name__ == "__main__":
    main()
