import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import TrainerCallback

df_case = pd.read_csv("/Users/tyj/Desktop/ELEC 576/Final Project Datasets/mimic_dataset_600.csv")
df_kw   = pd.read_csv("/Users/tyj/Desktop/ELEC 576/Final Project Datasets/mimic_dataset_600_keywords.csv")


df = df_case.merge(df_kw, on="HADM_ID", how="inner")


df_train = pd.DataFrame({
    "input":  df["Clinical_Notes"].astype(str),
    "output": df["Standard_Output"].astype(str)
})

dataset = Dataset.from_pandas(df_train)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

def preprocess(example):
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

    full_text = prompt + target

    tokens = tokenizer(full_text, truncation=True, max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_ds = dataset.map(preprocess, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./med_extractor_ckpt",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=50,

    logging_dir="./runs/med_extractor",
    report_to="tensorboard",              # enable TensorBoard
    logging_strategy="steps",

    save_strategy="no",
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds)
trainer.train()

trainer.save_model("./med_extractor_final")
tokenizer.save_pretrained("./med_extractor_final")

print("Successfully saved")