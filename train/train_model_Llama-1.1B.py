import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# ================= Configuration =================
CASE_DATA_PATH = "./mimic_dataset_600.csv"
KEYWORD_DATA_PATH = "./mimic_dataset_600_keywords.csv"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./med_extractor_final"
CHECKPOINT_DIR = "./med_extractor_ckpt"
# ===============================================

def print_status(step, message):
    print(f"\n{'='*10} Step {step}: {message} {'='*10}")

def format_prompt(example, tokenizer):
    """
    Formats the input clinical note and target output into a prompt structure.
    """
    note = example["input"]
    target = example["output"]
    
    prompt = (
        "You are a medical information extraction assistant. "
        "Given the following clinical note, extract key information.\n\n"
        "Format your answer exactly as:\n"
        "DISEASE:\n- ...\nSYMPTOMS:\n- ...\nTREATMENT:\n- ...\n\n"
        "CLINICAL NOTE:\n"
        f"{note}\n\n"
        "ANSWER:\n"
    )
    
    full_text = prompt + target + tokenizer.eos_token
    
    # Tokenize with fixed length padding/truncation
    tokens = tokenizer(
        full_text, 
        truncation=True, 
        max_length=256, 
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def main():
    # 1. Environment Check
    print_status(1, "Environment Setup")
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not detected. Defaulting to CPU.")

    if not os.path.exists(CASE_DATA_PATH) or not os.path.exists(KEYWORD_DATA_PATH):
        raise FileNotFoundError("One or more input CSV files are missing.")

    # 2. Data Loading & Merging
    print_status(2, "Loading Data")
    try:
        df_case = pd.read_csv(CASE_DATA_PATH)
        df_kw = pd.read_csv(KEYWORD_DATA_PATH)
        
        # Inner join on Hospital Admission ID
        df = df_case.merge(df_kw, on="HADM_ID", how="inner")
        print(f"Data merged successfully. Total samples: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare dataset format
    df_train = pd.DataFrame({
        "input": df["Clinical_Notes"].astype(str),
        "output": df["Standard_Output"].astype(str)
    })
    dataset = Dataset.from_pandas(df_train)

    # 3. Model Initialization
    print_status(3, f"Loading Model ({MODEL_NAME})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Ensure pad token is set (TinyLlama/Llama usually needs this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Apply LoRA Configuration
    print("-> Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("LoRA Model ready.")

    # 4. Preprocessing
    print_status(4, "Tokenizing Data")
    tokenized_ds = dataset.map(
        lambda x: format_prompt(x, tokenizer), 
        remove_columns=dataset.column_names
    )

    # 5. Training Setup
    print_status(5, "Configuring Trainer")
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_ds
    )

    # 6. Training Loop
    print_status(6, "Starting Training")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"\nTraining complete. Duration: {(end_time - start_time)/60:.2f} minutes")

    # 7. Save Adapters
    print_status(7, "Saving Model")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapters saved to {OUTPUT_DIR}")

    # 8. Visualization
    print_status(8, "Plotting Training Loss")
    try:
        history = trainer.state.log_history
        losses = [entry["loss"] for entry in history if "loss" in entry]
        steps = [entry["step"] for entry in history if "loss" in entry]

        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="Training Loss (LoRA)", color="green", marker="o")
            plt.title(f"Training Loss Curve ({MODEL_NAME})")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()
            plt.savefig("./training_loss_lora.png")
            print("Loss plot saved: training_loss_lora.png")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

    print("\nProcess Finished Successfully.")

if __name__ == "__main__":
    main()