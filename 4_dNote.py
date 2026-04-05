"""
KG-Importance Graph Embeddings for BHC (NO SBERT, NO external downloads)

What this code does (high-level):
- Uses your CSV (row-wise section→BHC similarity scores) as the *importance signal*.
- Builds a REAL section-section graph from correlations between section-importance scores across patients.
- Learns a GNN that turns per-patient section-importance scores into per-patient *node embeddings*.
- Injects those graph-derived node embeddings into LLaMA token embeddings (section-aware + importance-aware).
- Trains with LoRA + (GNN + projector + gate).
- Saves everything needed for inference (LoRA adapter, tokenizer, gnn/projector/gate, graph tensors).

Key: "Section importance should be KG graph embeddings"
✅ Here, importance lives inside the per-patient GNN node embeddings, not just a scalar.

Requirements:
- torch, transformers, peft, datasets, pandas, numpy
- torch_geometric (install on Colab if needed)

Colab install (run once):
!pip -q install transformers peft datasets bitsandbytes accelerate pandas numpy
# torch-geometric install depends on CUDA/torch version; use official instructions if needed:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
"""

import os
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# torch_geometric
from torch_geometric.nn import GCNConv

# -----------------------------
# 0. Registry & Config
# -----------------------------
SECTION_REGISTRY = {
    "PAD": 0, "Sex": 1, "Department_Service": 2, "Chief_Complaint": 3,
    "History_Present_Illness": 4, "Past_Medical_Surgical_History": 5,
    "Family_History": 6, "Personal_Social_History": 7, "Drug_History_Allergies": 8,
    "Clinical_Examination": 9, "Investigations": 10, "Treatment_During_Stay": 11,
    "Procedures_Surgeries_Done": 12, "Final_Diagnosis": 13,
    "Condition_at_Discharge": 14, "Discharge_Medications": 15,
    "Discharge_Disposition": 16, "Followup_Instructions": 17
}
SECTIONS = [k for k in SECTION_REGISTRY.keys() if k != "PAD"]
NUM_SECTIONS = len(SECTIONS)
ASSISTANT_TRIGGER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


@dataclass
class CFG:
    # Paths
    base_model: str = "/home/models/Llama-3.2-3B-Instruct"
    train_jsonl: str = "train_old_indian_standard.jsonl"
    val_jsonl: str = "valid_old_indian_standard.jsonl"
    scores_csv: str = "section_similarity_rowwise_allmetrics1.csv"   # row-wise scores

    output_dir: str = "results_da_kg_graph5000" #change da to bhc for hospital course training
    save_dir: str = "llama-3.2-3b-da-kg-graph-lora-5000" ##change da to bhc for hospital course training

    # Training
    seed: int = 42
    lr: float = 1e-4
    batch_size: int = 2
    grad_accum: int = 4
    epochs: int = 3
    max_len: int = 2048

    # GNN/Injection
    gnn_hidden: int = 64
    kg_embed_dim: int = 128     # node embedding dim from GNN
    corr_threshold: float = 0.10  # graph edge threshold on |corr|
    max_edges_per_node: int = 8   # keep graph sparse

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05


# -----------------------------
# 1. Build importance score lookup + section graph from CSV (NO SBERT)
# -----------------------------
def load_scores_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Expect columns like: note_id, subject_id, Sex__bhc_sbert, ...
    # We'll build a clean table with key + per-section score columns.
    # If your id column name differs, edit here:
    key_col = "note_id" if "note_id" in df.columns else ("subject_id" if "subject_id" in df.columns else None)
    if key_col is None:
        raise ValueError("CSV must contain 'note_id' or 'subject_id' as a key to align scores to samples.")

    # Build section score cols list
    score_cols = []
    for s in SECTIONS:
        col = f"{s}__da_sbert"#change da to bhc for hospital course training
        if col in df.columns:
            score_cols.append(col)
        else:
            # missing columns are allowed; we'll fill later with 0
            pass

    keep = [key_col] + score_cols
    out = df[keep].copy()

    # Ensure all expected section cols exist
    for s in SECTIONS:
        col = f"{s}__da_sbert" #change da to bhc for hospital course training
        if col not in out.columns:
            out[col] = 0.0

    # Replace NaNs
    out = out.fillna(0.0)
    return out, key_col


def build_score_lookup(df_scores: pd.DataFrame, key_col: str) -> dict:
    """
    Returns dict: key -> torch.FloatTensor[NUM_SECTIONS]
    """
    lut = {}
    for _, row in df_scores.iterrows():
        key = str(row[key_col])
        vec = []
        for s in SECTIONS:
            vec.append(float(row.get(f"{s}__da_sbert", 0.0)))
        lut[key] = torch.tensor(vec, dtype=torch.float32)
    return lut


def build_section_graph_from_corr(df_scores: pd.DataFrame, corr_threshold: float, max_edges_per_node: int):
    """
    Build section-section graph from correlation of per-patient section scores.
    Nodes = sections.
    Edge weight = |corr(i,j)|.
    Keeps top-k neighbors per node.
    Adds self-loops with weight 1.0.

    Returns:
      edge_index: [2, E] long
      edge_weight: [E] float
    """
    # matrix shape [N_patients, NUM_SECTIONS]
    mat = []
    for s in SECTIONS:
        mat.append(df_scores[f"{s}__da_sbert"].astype(float).values)
    mat = np.stack(mat, axis=1)  # [N, S]

    # corr matrix [S, S]
    corr = np.corrcoef(mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Build sparse edges: for each node i, pick top neighbors j by |corr|
    edges = set()
    weights = {}
    S = NUM_SECTIONS
    for i in range(S):
        # candidate js excluding i
        vals = [(j, abs(corr[i, j])) for j in range(S) if j != i]
        vals.sort(key=lambda x: x[1], reverse=True)
        kept = 0
        for j, w in vals:
            if w < corr_threshold:
                break
            # keep top-k per node
            edges.add((i, j))
            weights[(i, j)] = float(w)
            kept += 1
            if kept >= max_edges_per_node:
                break

    # Make undirected (optional but usually good)
    undirected_edges = set()
    for (i, j) in edges:
        undirected_edges.add((i, j))
        undirected_edges.add((j, i))
        weights[(j, i)] = weights[(i, j)]

    # Add self-loops
    for i in range(S):
        undirected_edges.add((i, i))
        weights[(i, i)] = 1.0

    edge_list = sorted(list(undirected_edges))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([weights[e] for e in edge_list], dtype=torch.float32)
    return edge_index, edge_weight


# -----------------------------
# 2. GNN that converts per-patient section scores -> per-patient node embeddings
# -----------------------------
class SectionImportanceGCN(nn.Module):
    """
    Input node features per patient: x_p is [NUM_SECTIONS, 1] (the section score)
    Graph is fixed across patients (edge_index, edge_weight).
    Output node embeddings per patient: [NUM_SECTIONS, kg_embed_dim]
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        # x: [B*S, 1]
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        return h  # [B*S, out_dim]


def batch_graph(edge_index, edge_weight, batch_size: int, num_nodes: int, device):
    """
    Repeat a single graph for B examples by offsetting node indices.
    Returns batched edge_index, edge_weight.
    """
    # edge_index: [2, E]
    E = edge_index.size(1)
    # offsets: [B]
    offsets = torch.arange(batch_size, device=device, dtype=torch.long) * num_nodes  # [B]
    # expand edges for each batch
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    ei_batched = []
    ew_batched = []
    for b in range(batch_size):
        ei_batched.append(ei + offsets[b])
        ew_batched.append(ew)
    ei_batched = torch.cat(ei_batched, dim=1)  # [2, B*E]
    ew_batched = torch.cat(ew_batched, dim=0)  # [B*E]
    return ei_batched, ew_batched


# -----------------------------
# 3. LLaMA wrapper: inject graph-derived node embeddings (importance lives here)
# -----------------------------
class LlamaWithGraphKG(nn.Module):
    """
    - section_scores (per sample) -> GNN -> node embeddings
    - tokens use section_ids to gather node embeddings
    - projector maps node-embed -> llama hidden
    - learned gate controls strength (but gate is computed from node embeddings => still "graph embedding driven")
    """
    def __init__(self, llama, edge_index, edge_weight, gnn_hidden, kg_embed_dim, llama_dim):
        super().__init__()
        self.llama = llama

        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.gnn = SectionImportanceGCN(hidden_dim=gnn_hidden, out_dim=kg_embed_dim)

        self.projector = nn.Sequential(
            nn.Linear(kg_embed_dim, llama_dim),
            nn.SiLU(),
            nn.Linear(llama_dim, llama_dim),
        )

        # Gate derived from graph embeddings (not from section id)
        self.gate = nn.Sequential(
            nn.Linear(kg_embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, section_ids, section_scores, attention_mask=None, labels=None, **kwargs):
        """
        input_ids: [B,T]
        section_ids: [B,T] values in [0..17]
        section_scores: [B, NUM_SECTIONS] scores aligned to SECTIONS list (no PAD)
        """
        B, T = input_ids.shape
        device = input_ids.device
        inputs_embeds = self.llama.get_input_embeddings()(input_ids)

        # Build per-patient node features: [B, S, 1]
        # If you want PAD node too, you can add it; but we keep only real sections.
        x = section_scores.to(device).unsqueeze(-1)  # [B,S,1]
        x = x.reshape(B * NUM_SECTIONS, 1)          # [B*S,1]

        # Batched graph
        ei, ew = batch_graph(self.edge_index, self.edge_weight, B, NUM_SECTIONS, device)

        # Run GNN => per-node embeddings
        node_emb = self.gnn(x, ei, ew)                               # [B*S, kg_dim]
        node_emb = node_emb.reshape(B, NUM_SECTIONS, -1)             # [B,S,kg_dim]

        # Gather per-token graph embedding using section_ids
        # section_ids includes PAD=0 plus section ids 1..17.
        # Our node_emb is indexed by SECTIONS (1..17). We need to map:
        # - PAD(0) -> zero vector
        # - id k (1..17) -> node_emb[:, k-1, :]
        kg_dim = node_emb.size(-1)
        pad_vec = torch.zeros((B, 1, kg_dim), device=device, dtype=node_emb.dtype)
        table = torch.cat([pad_vec, node_emb], dim=1)  # [B, 1+S, kg_dim] => indices match SECTION_REGISTRY ids
        kg_vecs = table.gather(1, section_ids.unsqueeze(-1).expand(-1, -1, kg_dim))  # [B,T,kg_dim]

        kg_proj = self.projector(kg_vecs)  # [B,T,H]
        gate = self.gate(kg_vecs)          # [B,T,1]

        # dtype match
        if kg_proj.dtype != inputs_embeds.dtype:
            kg_proj = kg_proj.to(inputs_embeds.dtype)
        if gate.dtype != inputs_embeds.dtype:
            gate = gate.to(inputs_embeds.dtype)

        combined = inputs_embeds + gate * kg_proj

        return self.llama(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def save_all(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.edge_index.cpu(), os.path.join(save_dir, "kg_edge_index.pt"))
        torch.save(self.edge_weight.cpu(), os.path.join(save_dir, "kg_edge_weight.pt"))
        torch.save(self.gnn.state_dict(), os.path.join(save_dir, "kg_gnn.pt"))
        torch.save(self.projector.state_dict(), os.path.join(save_dir, "kg_projector.pt"))
        torch.save(self.gate.state_dict(), os.path.join(save_dir, "kg_gate.pt"))
        # LoRA weights etc.
        self.llama.save_pretrained(save_dir)


# -----------------------------
# 4. Data processing: build section_ids + labels + section_scores
# -----------------------------
def build_section_ids_and_input(ex, tokenizer, max_len: int):
    """
    Returns: input_ids(list[int]), section_ids(list[int]), labels(list[int])
    Labels are -100 for everything before assistant content.
    """
    input_ids = []
    section_ids = []

    # system
    sys_txt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnalyze clinical data, Write discharge advice.<|eot_id|>"
    sys_ids = tokenizer.encode(sys_txt, add_special_tokens=False)
    input_ids += sys_ids
    section_ids += [0] * len(sys_ids)

    # sections (user content)
    for sec_name, sec_id in SECTION_REGISTRY.items():
        if sec_name == "PAD":
            continue
        content = str(ex.get(sec_name, ""))
        if not content or content.lower() == "nan" or not content.strip():
            continue
        chunk = f"<{sec_name}> {content}\n"
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        input_ids += ids
        section_ids += [sec_id] * len(ids)

    # assistant target
    tgt = f"{ASSISTANT_TRIGGER}{ex.get('Discharge_Advice','')}<|eot_id|>" #change Discharge_Advice to Hospital_Course for hospital course training
    tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)

    assistant_start = len(input_ids)
    input_ids += tgt_ids
    section_ids += [0] * len(tgt_ids)

    # truncate
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        section_ids = section_ids[:max_len]

    # labels: only assistant tokens contribute
    labels = [-100] * min(assistant_start, len(input_ids))
    labels += input_ids[assistant_start:len(input_ids)]
    labels = labels[:len(input_ids)]

    return input_ids, section_ids, labels


def preprocess_with_scores(examples, tokenizer, score_lut: dict, key_field: str, max_len: int):
    batch_input_ids = []
    batch_section_ids = []
    batch_labels = []
    batch_scores = []

    n = len(examples[key_field]) if key_field in examples else len(examples["Sex"])

    # best-effort: figure key per example
    for i in range(n):
        # create a dict-like row for build_section_ids_and_input
        row = {k: examples[k][i] if k in examples else "" for k in examples.keys()}

        input_ids, section_ids, labels = build_section_ids_and_input(row, tokenizer, max_len)

        # section scores vector
        key = None
        if key_field in examples:
            key = str(examples[key_field][i])
        elif "note_id" in examples:
            key = str(examples["note_id"][i])
        elif "subject_id" in examples:
            key = str(examples["subject_id"][i])

        if key is not None and key in score_lut:
            scores = score_lut[key]
        else:
            scores = torch.zeros(NUM_SECTIONS, dtype=torch.float32)

        batch_input_ids.append(input_ids)
        batch_section_ids.append(section_ids)
        batch_labels.append(labels)
        batch_scores.append(scores.tolist())

    return {
        "input_ids": batch_input_ids,
        "section_ids": batch_section_ids,
        "labels": batch_labels,
        "section_scores": batch_scores,  # [S] float list
    }


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features):
        batch_in  = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_sec = [torch.tensor(f["section_ids"], dtype=torch.long) for f in features]
        batch_lab = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        batch_sc  = [torch.tensor(f["section_scores"], dtype=torch.float32) for f in features]  # [S]

        batch_in  = torch.nn.utils.rnn.pad_sequence(batch_in,  batch_first=True, padding_value=pad_id)
        batch_sec = torch.nn.utils.rnn.pad_sequence(batch_sec, batch_first=True, padding_value=0)
        batch_lab = torch.nn.utils.rnn.pad_sequence(batch_lab, batch_first=True, padding_value=-100)

        batch_sc = torch.stack(batch_sc, dim=0)  # [B,S]

        attn = (batch_in != pad_id).long()
        return {
            "input_ids": batch_in,
            "section_ids": batch_sec,
            "labels": batch_lab,
            "attention_mask": attn,
            "section_scores": batch_sc
        }

    return collate


# -----------------------------
# 5. Custom Trainer
# -----------------------------
class GraphTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            section_ids=inputs["section_ids"],
            section_scores=inputs["section_scores"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save graph parts + LoRA adapter
        self.model.save_all(output_dir)
        self.processing_class.save_pretrained(output_dir)


# -----------------------------
# 6. Main training
# -----------------------------
def main_train():
    cfg = CFG()
    set_seed(cfg.seed)

    # 1) Load scores + build lookup + graph
    df_scores, csv_key = load_scores_table(cfg.scores_csv)
    score_lut = build_score_lookup(df_scores, csv_key)
    edge_index, edge_weight = build_section_graph_from_corr(
        df_scores, corr_threshold=cfg.corr_threshold, max_edges_per_node=cfg.max_edges_per_node
    )
    print(f"[KG] Graph edges: {edge_index.size(1)}")

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [f"<{k}>" for k in SECTIONS]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # 3) Dataset
    raw = load_dataset("json", data_files={"train": cfg.train_jsonl, "val": cfg.val_jsonl})

    # optional subset for quick test
    raw["train"] = raw["train"].select(range(min(5000, len(raw["train"]))))
    raw["val"]   = raw["val"].select(range(min(500, len(raw["val"]))))

    # choose key_field for dataset alignment (best-effort)
    key_field = "note_id" if "note_id" in raw["train"].column_names else ("subject_id" if "subject_id" in raw["train"].column_names else None)
    if key_field is None:
        # still works, but scores will be zero unless you modify preprocess_with_scores to match your id field
        key_field = "Sex"  # dummy to keep code running
        print("⚠️ Could not find note_id/subject_id in JSONL columns. Scores lookup may be all zeros. Add note_id to JSONL for correct alignment.")

    tokenized = raw.map(
        lambda x: preprocess_with_scores(x, tokenizer, score_lut, key_field, cfg.max_len),
        batched=True,
        remove_columns=raw["train"].column_names
    )

    # 4) Load base model 4-bit + LoRA
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb, device_map="auto")
    base.resize_token_embeddings(len(tokenizer))

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, peft_cfg)

    # 5) Wrap with graph-KG injection
    model = LlamaWithGraphKG(
        llama=base,
        edge_index=edge_index,
        edge_weight=edge_weight,
        gnn_hidden=cfg.gnn_hidden,
        kg_embed_dim=cfg.kg_embed_dim,
        llama_dim=base.config.hidden_size
    )

    # 6) Train
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = GraphTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=make_collator(tokenizer),
        processing_class=tokenizer
    )

    trainer.train()

    # 7) Final save (includes graph tensors + gnn/projector/gate + LoRA + tokenizer)
    os.makedirs(cfg.save_dir, exist_ok=True)
    model.save_all(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"✅ Saved everything to: {cfg.save_dir}")
    print("Files you should see: adapter_config.json, adapter_model.safetensors (or bin), tokenizer files, kg_edge_index.pt, kg_edge_weight.pt, kg_gnn.pt, kg_projector.pt, kg_gate.pt")


# -----------------------------
# 7. Minimal inference (greedy loop) using saved graph-KG adapter
# -----------------------------
@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, section_ids, section_scores, max_new_tokens=200):
    device = input_ids.device
    attn = torch.ones_like(input_ids, device=device)
    cur_in = input_ids
    cur_sec = section_ids

    for _ in range(max_new_tokens):
        out = model(
            input_ids=cur_in,
            section_ids=cur_sec,
            section_scores=section_scores,
            attention_mask=attn,
            labels=None
        )
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
        if nxt.item() == tokenizer.eos_token_id:
            break
        cur_in = torch.cat([cur_in, nxt], dim=1)
        cur_sec = torch.cat([cur_sec, torch.zeros((1, 1), dtype=cur_sec.dtype, device=device)], dim=1)
        attn = torch.ones_like(cur_in, device=device)

    return tokenizer.decode(cur_in[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def prepare_single_input(sample: dict, tokenizer):
    input_ids = []
    section_ids = []

    sys_txt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnalyze clinical data, Write discharge advice.<|eot_id|>"
    sys_ids = tokenizer.encode(sys_txt, add_special_tokens=False)
    input_ids += sys_ids
    section_ids += [0] * len(sys_ids)

    for sec_name, sec_id in SECTION_REGISTRY.items():
        if sec_name == "PAD":
            continue
        content = str(sample.get(sec_name, ""))
        if not content or content.lower() == "nan" or not content.strip():
            continue
        chunk = f"<{sec_name}> {content}\n"
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        input_ids += ids
        section_ids += [sec_id] * len(ids)

    trig_ids = tokenizer.encode(ASSISTANT_TRIGGER, add_special_tokens=False)
    input_ids += trig_ids
    section_ids += [0] * len(trig_ids)

    return torch.tensor([input_ids], dtype=torch.long), torch.tensor([section_ids], dtype=torch.long)



# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    # TRAIN:
    main_train()
