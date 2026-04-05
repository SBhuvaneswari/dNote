# train_adapter_di_jsonl_trl.py
# DI (Discharge Advice/Instructions) LoRA adapter training on JSONL using TRL SFTTrainer

import gc
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTTrainer, SFTConfig

torch.cuda.empty_cache()
gc.collect()


# -----------------------------
# 1. Config
# -----------------------------
@dataclass
class TrainingConfig:
    model_name: str = "/home/models/Llama-3.2-3B-Instruct"
    #model_name: str = "/home/models/Phi-3.5-Mini-Instruct"
    train_jsonl: str = "train_old_indian_standard.jsonl"
    val_jsonl: str = "valid_old_indian_standard.jsonl"

    output_dir_di: str = "results_di_5000"
    adapter_save_dir_di: str = "llama-3.5-3.8B-sep-v1-di_5000"

    seed: int = 42
    learning_rate: float = 5e-5
    batch_size: int = 2
    grad_accumulation: int = 8
    num_epochs: int = 3

    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 2

    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_train_samples: int = 5000
    max_eval_samples: int = 1000   

# -----------------------------
# 2. Reproducibility
# -----------------------------
def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 3. Dataset loading
# -----------------------------
def load_datasets(train_path: str, val_path: str) -> Dict[str, Any]:
    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds = load_dataset("json", data_files=val_path, split="train")
    return {"train": train_ds, "eval": val_ds}


# -----------------------------
# 4. Features + Prompt builder
# -----------------------------
def get_input_features() -> List[str]:
    return [
        "Sex", "Department_Service",
        "Chief_Complaint", "History_Present_Illness", "Past_Medical_Surgical_History",
        "Family_History", "Personal_Social_History", "Drug_History_Allergies",
        "Clinical_Examination", "Investigations", "Treatment_During_Stay",
        "Procedures_Surgeries_Done", "Final_Diagnosis",
        "Condition_at_Discharge", "Discharge_Medications", "Discharge_Disposition",
        "Followup_Instructions",
    ]


def build_text_for_di(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert one JSONL row into one training text string for DI in Llama-3 chat format.
    IMPORTANT: Your JSONL must contain DI target column.
    Common possibilities: "Discharge_Advice" OR "Followup_Instructions".
    This code tries multiple keys safely.
    """

    input_features = get_input_features()

    BOS = "<|begin_of_text|>"
    EOT = "<|eot_id|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"

    system_msg = (
        "You are an expert medical AI assistant for Indian hospitals. "
        "Write clear, safe, and patient-friendly discharge instructions. "
        "Do not invent medications, dosages, dates, or follow-up visits."
    )

    patient_data_str = ""
    for feature in input_features:
        val = example.get(feature, None)
        val_str = (
            str(val).strip()
            if val is not None and str(val).lower() != "nan"
            else "Not Recorded"
        )
        patient_data_str += f"{feature.replace('_', ' ')}: {val_str}\n"

    instruction = (
        "Based on the patient data provided, write the 'Discharge Instructions / Discharge Advice'for the patient."
    )

    # ---- DI TARGET FIELD SELECTION ----
    # Prefer Discharge_Advice if you have it. Else fallback to Followup_Instructions.
    # You can add more keys if your dataset uses different column name.
    target_keys = ["Discharge_Advice", "Discharge_Instructions", "DI",]
    target = ""
    for k in target_keys:
        if k in example and example[k] is not None and str(example[k]).strip() != "":
            target = str(example[k]).strip()
            break

    # If still empty, keep a placeholder (better than crashing)
    if target == "":
        target = "Not Recorded"

    text = (
        f"{BOS}{START_HEADER}system{END_HEADER}\n\n{system_msg}{EOT}"
        f"{START_HEADER}user{END_HEADER}\n\n"
        f"Patient Data:\n{patient_data_str}\n\n"
        f"Task: {instruction}{EOT}"
        f"{START_HEADER}assistant{END_HEADER}\n\n"
        f"{target}{EOT}"
    )

    return {"text": text}


# -----------------------------
# 5. Model + tokenizer (4-bit)
# -----------------------------
def load_model_and_tokenizer(model_name: str):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# -----------------------------
# 6. LoRA + Trainer
# -----------------------------
def setup_lora_config(cfg: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def create_trainer(cfg: TrainingConfig, model, tokenizer, datasets: Dict[str, Any]) -> SFTTrainer:
    peft_config = setup_lora_config(cfg)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir=cfg.output_dir_di,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        optim="paged_adamw_32bit",
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=cfg.seed,

        # evaluation / saving
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,

        fp16=not use_bf16,
        bf16=use_bf16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        packing=False,
        report_to = "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        processing_class=tokenizer,
        
    )
    return trainer


# -----------------------------
# 7. Main
# -----------------------------
def main():
    cfg = TrainingConfig()
    set_reproducibility(cfg.seed)

    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected. Training will be very slow.")

    print("Loading datasets (jsonl)...")
    datasets = load_datasets(cfg.train_jsonl, cfg.val_jsonl)

    # print("Building 'text' field for DI training...")
    # datasets["train"] = datasets["train"].map(build_text_for_di, remove_columns=[])
    # datasets["eval"] = datasets["eval"].map(build_text_for_di, remove_columns=[])
    if cfg.max_train_samples is not None:
        n = min(cfg.max_train_samples, len(datasets["train"]))
        datasets["train"] = datasets["train"].select(range(n))

    if cfg.max_eval_samples is not None:
        n = min(cfg.max_eval_samples, len(datasets["eval"]))
        datasets["eval"] = datasets["eval"].select(range(n))

    print("✅ Train rows:", len(datasets["train"]))
    print("✅ Val rows:", len(datasets["eval"]))
    print("✅ Example[0] text chars:", len(datasets["train"][0]["text"]))

    print("\n=== Training DI adapter (Discharge Instructions) ===")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name)
    trainer = create_trainer(cfg, model, tokenizer, datasets)

    trainer.train()

    print(f"\nSaving adapter to: {cfg.adapter_save_dir_di}")
    trainer.model.save_pretrained(cfg.adapter_save_dir_di)
    tokenizer.save_pretrained(cfg.adapter_save_dir_di)

    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n✅ Finished DI adapter training.")


if __name__ == "__main__":
    main()
