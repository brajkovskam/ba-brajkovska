#!/usr/bin/env python
"""
Enhanced BERT Inference Script with Advanced Performance Metrics

This script tests a fine-tuned BERT classification model (e.g., for paraphrase detection)
on 50 sentence pairs. It runs inference for variable batch sizes and logs the following metrics:

  - TTFT (Time To First Token): The time from input start (tokenization) until model logits are available.
  - ITL (Inter-Token Latency): A simulated sequential processing delay between each output token.
  - TPS (Tokens Per Second): Estimated as total input tokens processed divided by the inference time.
  - Latency: The full elapsed time (from input tokenization through post‑processing) per batch.
  - QPS (Queries Per Second): Calculated as the number of samples in the batch divided by the batch latency.

Test scenarios: A predefined list of 50 sentence pairs is used.
Usage:
    python enhanced_inference.py --model_path /path/to/fine_tuned_model --max_length 128 --noise_level 0.0
"""

import time
import torch
import torch.nn.functional as F
import argparse
import logging
import sys
import random
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Logging Setup
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s", 
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# Device Configuration: mps, CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Test Scenarios: 50 Sentence Pairs with a unique 'id' and paraphrase labels (1: paraphrase, 0: non-paraphrase)
sample_data = [
    {"id": 1,  "text1": "The weather is nice today.", "text2": "It's a beautiful day outside.", "label": 1},
    {"id": 2,  "text1": "I love pizza.", "text2": "I hate pizza.", "label": 0},
    {"id": 3,  "text1": "She is a talented musician.", "text2": "Her musical skills are impressive.", "label": 1},
    {"id": 4,  "text1": "The sky is blue.", "text2": "The grass is green.", "label": 0},
    {"id": 5,  "text1": "I enjoy reading books.", "text2": "Reading is one of my hobbies.", "label": 1},
    {"id": 6,  "text1": "She went to the store.", "text2": "He stayed home.", "label": 0},
    {"id": 7,  "text1": "The restaurant has good food.", "text2": "The restaurant has excellent service.", "label": 0},
    {"id": 8,  "text1": "He is a doctor.", "text2": "He works in a hospital.", "label": 0},
    {"id": 9,  "text1": "She bought a new car.", "text2": "She sold her old car.", "label": 0},
    {"id": 10, "text1": "The cat is on the mat.", "text2": "The cat is not on the mat.", "label": 0},
    {"id": 11, "text1": "I will go to the party.", "text2": "I might go to the party.", "label": 0},
    {"id": 12, "text1": "The meeting is at 3 PM.", "text2": "The meeting is at 4 PM.", "label": 0},
    {"id": 13, "text1": "All dogs are mammals.", "text2": "Some dogs are mammals.", "label": 0},
    {"id": 14, "text1": "No one likes rain.", "text2": "Everyone likes rain.", "label": 0},
    {"id": 15, "text1": "She always arrives on time.", "text2": "She sometimes arrives on time.", "label": 0},
    {"id": 16, "text1": "The bank is near the river.", "text2": "I need to go to the bank to withdraw money.", "label": 0},
    {"id": 17, "text1": "She saw a bat flying.", "text2": "He hit the ball with a bat.", "label": 0},
    {"id": 18, "text1": "The plant needs water.", "text2": "The factory plant is closing.", "label": 0},
    {"id": 19, "text1": "He kicked the bucket.", "text2": "He died.", "label": 1},
    {"id": 20, "text1": "She has a green thumb.", "text2": "She is good at gardening.", "label": 1},
    {"id": 21, "text1": "It's raining cats and dogs.", "text2": "There are many pets outside.", "label": 0},
    {"id": 22, "text1": "Although it was cold, he went outside.", "text2": "He went outside despite the cold.", "label": 1},
    {"id": 23, "text1": "Because she studied hard, she passed the test.", "text2": "She passed the test due to her hard studying.", "label": 1},
    {"id": 24, "text1": "If you heat water, it boils.", "text2": "Water boils when heated.", "label": 1},
    {"id": 25, "text1": "The quick brown fox jumps over the lazy dog.", "text2": "A fast fox leaps over a slow dog.", "label": 1},
    {"id": 26, "text1": "A penny saved is a penny earned.", "text2": "Saving money is as good as earning it.", "label": 1},
    {"id": 27, "text1": "Actions speak louder than words.", "text2": "What you do is more important than what you say.", "label": 1},
    {"id": 28, "text1": "The Earth is flat.", "text2": "The Earth is round.", "label": 0},
    {"id": 29, "text1": "Cats are mammals.", "text2": "Cats are reptiles.", "label": 0},
    {"id": 30, "text1": "Python is a programming language.", "text2": "Python is a snake.", "label": 0},
    {"id": 31, "text1": "The movie was thrilling.", "text2": "That film kept me on edge.", "label": 1},
    {"id": 32, "text1": "He runs every morning.", "text2": "He jogs every evening.", "label": 0},
    {"id": 33, "text1": "The coffee is too hot.", "text2": "The tea is too cold.", "label": 0},
    {"id": 34, "text1": "They traveled to Paris.", "text2": "They visited France’s capital.", "label": 1},
    {"id": 35, "text1": "I forgot my lines.", "text2": "I remembered my script.", "label": 0},
    {"id": 36, "text1": "The dog barked loudly.", "text2": "The dog made a loud noise.", "label": 1},
    {"id": 37, "text1": "She smiled at him.", "text2": "She frowned at him.", "label": 0},
    {"id": 38, "text1": "The sun sets in the west.", "text2": "The sun rises in the east.", "label": 0},
    {"id": 39, "text1": "He wrote a letter.", "text2": "He composed a note.", "label": 1},
    {"id": 40, "text1": "The book is on the shelf.", "text2": "The book is under the table.", "label": 0},
    {"id": 41, "text1": "It’s a sunny day.", "text2": "The weather is bright.", "label": 1},
    {"id": 42, "text1": "I lost my phone.", "text2": "I found my wallet.", "label": 0},
    {"id": 43, "text1": "The team won the game.", "text2": "The team lost the match.", "label": 0},
    {"id": 44, "text1": "She sings beautifully.", "text2": "Her singing is lovely.", "label": 1},
    {"id": 45, "text1": "The car is red.", "text2": "The truck is blue.", "label": 0},
    {"id": 46, "text1": "He arrived early.", "text2": "He showed up ahead of time.", "label": 1},
    {"id": 47, "text1": "The soup tastes salty.", "text2": "The dessert tastes sweet.", "label": 0},
    {"id": 48, "text1": "They danced all night.", "text2": "They partied until dawn.", "label": 0},
    {"id": 49, "text1": "The child laughed.", "text2": "The kid giggled.", "label": 1},
    {"id": 50, "text1": "The moon is full tonight.", "text2": "The stars are bright tonight.", "label": 0},
]

def add_noise(text, noise_level):
    """Add random noise to the input text by replacing characters."""
    if noise_level <= 0:
        return text
    num_noisy_chars = int(len(text) * noise_level)
    for _ in range(num_noisy_chars):
        pos = random.randint(0, len(text) - 1)
        text = text[:pos] + random.choice(string.ascii_letters) + text[pos+1:]
    return text

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced BERT Inference Script with Advanced Performance Metrics"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise level for input perturbation (0-1).")
    args = parser.parse_args()

    # Load model and tokenizer from the provided model path (single model scenario)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # Define batch sizes to simulate variable query loads
    batch_sizes = [8, 16, 24, 32, 64]
    metrics_summary = {}  # to hold summary metrics per batch size
    total_samples = len(sample_data)

    for bs in batch_sizes:
        logger.info(f"\n--- Running inference with batch size {bs} ---")
        total_time = 0.0          # cumulative time for all batches at this batch size
        total_tokens = 0          # cumulative input tokens processed
        total_ttft = 0.0          # cumulative time-to-first-token (TTFT)
        total_itl = 0.0           # cumulative inter-token latency (ITL)
        batch_count = 0

        # Process the test set in batches
        for i in range(0, total_samples, bs):
            batch = sample_data[i: i+bs]
            actual_batch_size = len(batch)
            batch_count += 1

            # Prepare batch inputs: add noise to each sentence as needed
            texts1 = [add_noise(item["text1"], args.noise_level) for item in batch]
            texts2 = [add_noise(item["text2"], args.noise_level) for item in batch]
            encoded = tokenizer(texts1, texts2, padding=True, truncation=True,
                                max_length=args.max_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in encoded.items()}

            # Start timing (includes tokenization and data transfer)
            start_time = time.perf_counter()
            outputs = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            mid_time = time.perf_counter()

            # TTFT: time until logits are first available
            ttft = mid_time - start_time

            # Count tokens processed (using the attention mask to omit padding)
            tokens_in_batch = int(inputs["attention_mask"].sum().item())
            total_tokens += tokens_in_batch

            # Simulate processing each output sequentially to get ITL (one output per sample)
            logits = outputs.logits
            inter_token_intervals = []
            prev_time = mid_time
            for j in range(actual_batch_size):
                _ = F.softmax(logits[j], dim=-1)
                current_time = time.perf_counter()
                inter_token_intervals.append(current_time - prev_time)
                prev_time = current_time

            end_time = time.perf_counter()
            latency = end_time - start_time  # total time for the batch

            # Compute average ITL (skip the first interval)
            if len(inter_token_intervals) > 1:
                avg_itl = sum(inter_token_intervals[1:]) / (len(inter_token_intervals) - 1)
            else:
                avg_itl = 0.0

            # TPS: estimated as total input tokens divided by TTFT (approximation)
            tps = tokens_in_batch / (ttft if ttft > 0 else latency)
            # QPS: queries (samples) per second in this batch
            qps_batch = actual_batch_size / latency

            logger.info(f"Batch {batch_count:>2} (size {actual_batch_size}): "
                        f"TTFT = {ttft*1000:.2f} ms, "
                        f"ITL = {avg_itl*1000:.2f} ms, "
                        f"Latency = {latency*1000:.2f} ms, "
                        f"TPS = {tps:.2f} tokens/s, "
                        f"QPS = {qps_batch:.2f} queries/s")
            
            total_ttft += ttft
            total_itl += avg_itl
            total_time += latency

        # Compute overall metrics for this batch size configuration
        overall_qps = total_samples / total_time if total_time > 0 else float('inf')
        overall_tps = total_tokens / total_time if total_time > 0 else float('inf')
        avg_ttft = total_ttft / batch_count if batch_count > 0 else 0.0
        avg_itl = total_itl / batch_count if batch_count > 0 else 0.0
        avg_latency = total_time / batch_count if batch_count > 0 else 0.0

        metrics_summary[bs] = {
            "avg_ttft_ms": avg_ttft * 1000,
            "avg_itl_ms": avg_itl * 1000,
            "avg_latency_ms": avg_latency * 1000,
            "tps": overall_tps,
            "qps": overall_qps
        }

    # Final Summary: display aggregated performance metrics across batch sizes
    logger.info("\n===== Performance Summary =====")
    for bs, metrics in metrics_summary.items():
        logger.info(f"Batch Size {bs}: "
                    f"Avg TTFT = {metrics['avg_ttft_ms']:.2f} ms, "
                    f"Avg ITL = {metrics['avg_itl_ms']:.2f} ms, "
                    f"Avg Latency = {metrics['avg_latency_ms']:.2f} ms, "
                    f"TPS = {metrics['tps']:.2f} tokens/s, "
                    f"QPS = {metrics['qps']:.2f} queries/s")

if __name__ == "__main__":
    main()
