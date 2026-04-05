import torch
import random
import numpy as np
import os
import gc
import wandb
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from transformers import set_seed

torch.cuda.empty_cache()
gc.collect()

@dataclass
class TrainingConfig:
    model_name: str = "/home/models/Llama-3.2-3B-Instruct"
    new_model_name: str = "Llama-3.2-comb-v1"
    train_csv: str = "train_old_indian_standard.csv"
    val_csv: str = "valid_old_indian_standard.csv"
    output_dir: str = "results_combined"

    seed: int = 42
    max_seq_length: int = 4096
    learning_rate: float =5e-5
    batch_size: int = 16
    grad_accumulation: int = 1
    num_epochs: int = 3
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05

def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_datasets(train_path: str, val_path: str) -> Dict[str, Any]:
    train_ds = load_dataset("csv", data_files=train_path, split="train")
    val_ds = load_dataset("csv", data_files=val_path, split="train")
    return {"train": train_ds, "eval": val_ds}

# def load_datasets(train_path: str, val_path: str) -> Dict[str, Any]:
#     # 1. Load the full dataset
#     full_train_ds = load_dataset("csv", data_files=train_path, split="train")
#     val_ds = load_dataset("csv", data_files=val_path, split="train")
    
#     # 2. Calculate size
#     subset_ratio = 0.2
#     subset_size = int(len(full_train_ds) * subset_ratio)
    
#     # 3. Shuffle and Select
#     train_ds = full_train_ds.shuffle(seed=42).select(range(subset_size))
    
#     print(f"--- Data Slicing Report ---")
#     print(f"Original Train Size: {len(full_train_ds)}")
#     print(f"Sliced Train Size ({int(subset_ratio*100)}%): {len(train_ds)}")
#     print(f"Validation Size: {len(val_ds)}")
#     print(f"---------------------------")

#     return {"train": train_ds, "eval": val_ds}

def get_input_features() -> List[str]:
    return [
        "Sex", "Department_Service",
        "Chief_Complaint", "History_Present_Illness", "Past_Medical_Surgical_History",
        "Family_History", "Personal_Social_History", "Drug_History_Allergies",
        "Clinical_Examination", "Investigations", "Treatment_During_Stay",
        "Procedures_Surgeries_Done", "Final_Diagnosis",
        "Condition_at_Discharge", "Discharge_Medications", "Discharge_Disposition","Followup_Instructions",
    ]

# --- ROBUST FORMATTING FUNCTION ---
def formatting_prompts_func(examples: Union[Dict, Any]) -> Union[List[str], str]:
    """
    Handles both single examples (dict) and batches (dict of lists).
    """
    input_features = get_input_features()
    
    # 1. Detect if input is a BATCH (List) or SINGLE ITEM
    is_batch = isinstance(examples["Hospital_Course"], list)
    
    # If it's a single item, wrap it in a list so we can use the same logic
    if not is_batch:
        examples = {k: [v] for k, v in examples.items()}

    output_texts = []
    
    # Llama 3 special tokens
    BOS = "<|begin_of_text|>"
    EOT = "<|eot_id|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    
    system_msg = (
        "You are an expert medical AI assistant. "
        "Analyze the patient clinical data. "
        "Generate a comprehensive 'Hospital Course' followed immediately by the 'Discharge Advice'. "
        "Use clear markdown headers."
    )

    for i in range(len(examples["Hospital_Course"])):
        # Build Input
        patient_data_str = ""
        for feature in input_features:
            val = examples[feature][i]
            val_str = str(val).strip() if val is not None and str(val).lower() != 'nan' else "N/A"
            patient_data_str += f"**{feature.replace('_', ' ')}**: {val_str}\n"

        # Build Target
        hc_text = str(examples["Hospital_Course"][i]).strip()
        da_text = str(examples["Discharge_Advice"][i]).strip()

        combined_response = (
            f"## Hospital Course\n{hc_text}\n\n"
            f"## Discharge Advice\n{da_text}"
        )

        full_text = (
            f"{BOS}{START_HEADER}system{END_HEADER}\n\n{system_msg}{EOT}"
            f"{START_HEADER}user{END_HEADER}\n\n"
            f"Patient Data:\n{patient_data_str}\n\n"
            f"Task: Write the Hospital Course and Discharge Advice.{EOT}"
            f"{START_HEADER}assistant{END_HEADER}\n\n"
            f"{combined_response}{EOT}"
        )
        output_texts.append(full_text)

    # 2. Return correct type based on input
    if is_batch:
        return output_texts  # Return List[str]
    else:
        return output_texts[0] # Return str (Fixes AttributeError)

def load_model_and_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map={"": 0}, 
        dtype=torch.float16,
        # attn_implementation="flash_attention_2"
    )
    
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def create_trainer(config: TrainingConfig, model, tokenizer, datasets: Dict[str, Any]) -> SFTTrainer:

    #layers are frozen correctly without freezing the adapter later
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    print("\n" + "="*30)
    model.print_trainable_parameters()
    print("="*30 + "\n")
    
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        optim="paged_adamw_32bit",
        save_steps=200, 
        logging_steps=20,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        # fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="cosine",
        seed=config.seed,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # max_seq_length=config.max_seq_length,
        packing=False,
        dataset_text_field="text",
        report_to="wandb",
        run_name=f"{config.new_model_name}-run",
        logging_dir="./logs",
    )
    
    return SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        formatting_func=formatting_prompts_func,
        args=training_args,
        processing_class=tokenizer,
    )

def main():
    os.environ["WANDB_API_KEY"] = "xyz" 
    os.environ["WANDB_PROJECT"] = "Llama-3.2-comb-v1"
    os.environ["WANDB_WATCH"] = "false"
    
    config = TrainingConfig()
    set_reproducibility(config.seed)
    
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Define checkpoint path based on output_dir
    # resume_ckpt = os.path.join(config.output_dir, "checkpoint-12900")
    # print(f"Resume checkpoint path: {resume_ckpt}")

    # if not os.path.exists(resume_ckpt):
    #     print("Checkpoint not found at this path. Training will start from scratch.")
    
    datasets = load_datasets(config.train_csv, config.val_csv)
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    
    trainer = create_trainer(config, model, tokenizer, datasets)
    
    #trainer.train()
    # 5. Train (resume if checkpoint exists)
    # print("Starting Combined HC/DA Training...")
    # if os.path.exists(resume_ckpt):
    #     print(f"Resuming training from checkpoint: {resume_ckpt}")
    #     trainer.train(resume_from_checkpoint=resume_ckpt)
    # else:
    #     print("No checkpoint found. Starting fresh training.")
    trainer.train()
    print(f"Saving to {config.new_model_name}...")
    trainer.model.save_pretrained(config.new_model_name)
    tokenizer.save_pretrained(config.new_model_name)
    print("✅ Done!")

if __name__ == "__main__":
    main()