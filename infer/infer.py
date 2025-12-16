from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import re
import torch

model_dir = "./med_extractor_final"  # directory where your fine-tuned model is saved

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

def extract_keywords(note_text: str) -> str:
    """
    Output: one keyword per line (mixed diseases/symptoms/treatments).
    """
    prompt = (
        "You are a medical keyword extraction assistant.\n"
        "Extract ONLY key medical keywords from the clinical note.\n"
        "Include diseases/diagnoses, key symptoms, and treatments/interventions.\n"
        "Rules:\n"
        "- Output ONE keyword per line\n"
        "- No numbering, no bullets, no section titles\n"
        "- Output keywords only\n\n"
        "CLINICAL NOTE:\n"
        f"{note_text}\n\n"
        "KEYWORDS:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ans = full_text.split("KEYWORDS:")[-1].strip() if "KEYWORDS:" in full_text else full_text.strip()

    lines: List[str] = []
    seen = set()

    for line in ans.splitlines():
        s = line.strip()
        if not s:
            continue

        if s.upper().rstrip(":") in {"KEYWORDS", "DISEASE", "DIAGNOSIS", "SYMPTOMS", "TREATMENT", "MEDICATIONS"}:
            continue

        s = re.sub(r"^\s*(\d+[\.\)]|\-|\•)\s*", "", s).strip()
        if not s:
            continue

        if s in {"- ...", "..."}:
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)

        lines.append(s)

    return "\n".join(lines)

if __name__ == "__main__":
    case = """
    A 65-year-old male with a history of hypertension presented with acute shortness of breath
    and hypoxemia. Chest X-ray showed bilateral pulmonary infiltrates. PaO2/FiO2 ratio was 90.
    He was diagnosed with acute respiratory distress syndrome (ARDS) and septic shock.
    Treatment included invasive mechanical ventilation, broad-spectrum antibiotics,
    norepinephrine infusion, and aggressive fluid resuscitation.
    """

    result = extract_keywords(case)
    print(result)
