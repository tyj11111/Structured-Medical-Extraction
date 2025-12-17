# Structured Medical Information Extraction

This project focuses on extracting structured medical information
from unstructured clinical notes using a fine-tuned Qwen language model.

The extracted information includes diseases, symptoms, and treatments.

This repository contains the training and inference code used
for a course project.



## What This Project Does

- Fine-tunes a Qwen Instruct model on MIMIC-style clinical data
- Extracts structured medical information from clinical notes:
  - Disease 
  - Symptoms
  - Medications
- Provides separate scripts for training and inference

This repository is organized as follows:
	•	train/
	•	train_model.py
	•	train_model_Llama-1.1B.py	
	
  Training script for fine-tuning the Qwen Instruct model on MIMIC clinical notes.
  It loads the dataset, constructs instruction-style prompts, and performs supervised fine-tuning.
  
	•	infer/
	•	infer.py
  Inference script for running the fine-tuned model on new clinical notes.
  It generates structured medical keywords including diseases, symptoms, and treatments.
  
	•	rag/
	•   rag.ipynb
	•	mimic_dataset_600_keywords.csv
  Contains experiments related to retrieval augmented generation, including embedding, retrieval, and reranking using clinical                 keyword data.


