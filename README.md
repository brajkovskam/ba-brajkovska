# ba-brajkovska

# README: BERT Model Setup and Experiment

This repository contains scripts for fine-tuning a BERT model on the MRPC dataset and performing inference with advanced performance metrics. Follow the instructions below to set up and run the experiments.

---

## Installation

1. **Install Dependencies**

   Ensure you have Python 3 installed, then install dependencies. For saving models with safetensors, run:

   ```bash
   python3 -m pip install torch numpy transformers datasets evaluate scikit-learn
   ```

2. **Install Other Requirements**

   Install the remaining Python packages (e.g., torch, transformers, datasets, evaluate, scikit-learn) as needed, preferably via a requirements file or manually.

---

## Training Script Overview

The training script (`train.py`) fine-tunes BERT on the MRPC dataset using the following key settings:

- **Hyperparameters:**
  - `LEARNING_RATE = 3e-5`
  - `BATCH_SIZE = 24`
  - `NUM_EPOCHS = 6`
  
- **Model and Tokenizer Setup:**
  - Uses `bert-base-cased` with 2 labels.
  - Saves the final model and tokenizer to the `final_model` folder.
  - If `safetensors` is installed, the weights are saved as `model.safetensors`; otherwise, as `pytorch_model.bin`.

- **Advanced Metrics:**
  - Measures TTFT (Time to First Training Step) during the first training step.
  - Logs training metrics like loss, TPS (Tokens per Second), and total training latency.

Refer to the comments in the script for details on each section.

---

## Inference Script Overview

The inference script (`enhanced_inference.py`) tests the fine-tuned model on 50 sentence pairs. It measures and logs:

- **TTFT (Time To First Token)**
- **ITL (Inter-Token Latency)**
- **TPS (Tokens Per Second)**
- **Latency (Per Batch)**
- **QPS (Queries Per Second)**

It evaluates performance under different batch sizes (e.g., 8, 16, 24, 32, 64) to simulate variable query loads.

Usage example:

```bash
python inference.py --model_path /path/to/fine_tuned_model --max_length 128 --noise_level 0.0
```

## Output
The output file after a successful training should look like output_file_

---




## Summary

This setup allows you to experiment with BERT fine-tuning and inference performance while ensuring that your model is saved correctly and performance metrics are logged. Make sure to update the scripts as described and verify your environment settings before running the experiments. Happy experimenting!
