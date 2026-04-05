# train_adapter_bhc_jsonl_trl.py
# BHC LoRA adapter training on JSONL using TRL SFTTrainer
# Fixes: KeyError(None), list.endswith error, indexing issues, max_seq_length mismatch

import os
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


# -----------------------------
# 0. Cleanup
# -----------------------------
torch.cuda.empty_cache()
gc.collect()


# -----------------------------
# 1. Config
# -----------------------------
@dataclass
class TrainingConfig:
    model_name: str = "/home/models/Llama-3.2-3B-Instruct"
    #model_name: str = "/home/models/Phi-3.5-Mini-Instruct"
    # JSONL paths (each line = one JSON object)
    train_jsonl: str = "train_old_indian_standard.jsonl"
    val_jsonl: str = "valid_old_indian_standard.jsonl"

    # Output
    output_dir_bhc: str = "results_bhc_5000"
    adapter_save_dir_bhc: str = "llama-3.2-3.8B-sep-v1-bhc_5000"

    # Training
    seed: int = 42
    learning_rate: float = 5e-5
    batch_size: int = 2              # keep small for 3B + 4bit; increase if GPU allows
    grad_accumulation: int = 8       # effective batch = batch_size * grad_accumulation
    num_epochs: int = 3

    # Logging / eval
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 2

    # LoRA
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
# 3. Dataset loading (JSONL)
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


def build_text_for_bhc(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert one JSONL row into one training text string in Llama-3 chat format.
    Returns {"text": "..."} which TRL can tokenize via dataset_text_field="text".
    """
    input_features = get_input_features()

    # Llama 3 chat tokens
    BOS = "<|begin_of_text|>"
    EOT = "<|eot_id|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"

    system_msg = (
        "You are an expert medical AI assistant for Indian hospitals. "
        "Analyze the patient clinical data carefully and produce factual, formal clinical hospital course."
        "Do not invent new information that are not in the patient data."
    )

    # Build patient data block
    patient_data_str = ""
    for feature in input_features:
        val = example.get(feature, None)
        val_str = (
            str(val).strip()
            if val is not None and str(val).lower() != "nan"
            else "Not Recorded"
        )
        patient_data_str += f"{feature.replace('_', ' ')}: {val_str}\n"

    instruction = "Based on the patient data provided, write the 'Hospital Course' summary."
    target = str(example.get("Hospital_Course", "")).strip()

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
# 6. LoRA config + Trainer
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

    # NOTE:
    # Some TRL versions use eval_strategy, some use evaluation_strategy.
    # SFTConfig in TRL newer versions uses eval_strategy.
    training_args = SFTConfig(
        output_dir=cfg.output_dir_bhc,
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

        # precision
        fp16=not use_bf16,
        bf16=use_bf16,

        # memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text", 

        # IMPORTANT: we will provide dataset_text_field="text"
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        processing_class=tokenizer,     # correct for newer TRL
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

    # # Build 'text' column (CRITICAL FIX)
    # print("Building 'text' field for BHC training...")
    # datasets["train"] = datasets["train"].map(build_text_for_bhc, remove_columns=[])
    # datasets["eval"]  = datasets["eval"].map(build_text_for_bhc, remove_columns=[])
    # ✅ Take first 5000 samples 
    if cfg.max_train_samples is not None:
        n = min(cfg.max_train_samples, len(datasets["train"]))
        datasets["train"] = datasets["train"].select(range(n))

    if cfg.max_eval_samples is not None:
        n = min(cfg.max_eval_samples, len(datasets["eval"]))
        datasets["eval"] = datasets["eval"].select(range(n))

    # Quick sanity checks
    print("✅ Train rows:", len(datasets["train"]))
    print("✅ Val rows:", len(datasets["eval"]))
    ex0 = datasets["train"][0]["text"]
    print("✅ Example[0] text chars:", len(ex0))

    print("\n=== Training BHC adapter (Hospital Course) ===")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name)
    trainer = create_trainer(cfg, model, tokenizer, datasets)

    trainer.train()

    print(f"\nSaving adapter to: {cfg.adapter_save_dir_bhc}")
    trainer.model.save_pretrained(cfg.adapter_save_dir_bhc)
    tokenizer.save_pretrained(cfg.adapter_save_dir_bhc)

    # cleanup
    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n✅ Finished BHC adapter training.")


if __name__ == "__main__":
    main()
