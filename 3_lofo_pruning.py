import json
import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
CSV_PATH = "section_similarity_mean_allmetrics_new.csv"
INPUT_JSONL = "testphase1c.jsonl"
FINAL_OUTPUT = "final_lofo_ablation_borda.jsonl"

BASE_MODEL_PATH = "/home/models/Llama-3.2-3B-Instruct"
BHC_ADAPTER_PATH = "/home/saranya/llm_project/train_base/Llama-3.2-3B-sep-v1-bhc"
DI_ADAPTER_PATH = "/home/saranya/llm_project/train_base/Llama-3.2-3B-sep-v1-di"

DEVICE_INDEX = 0
SAMPLE_SIZE = 20
TOP_K = 15

# Columns layout assumption:
# col0 = feature name
# col1..col5  = 5 metrics for BHC
# col6..col10 = 5 metrics for DI
BHC_COL_SLICE = slice(1, 6)
DI_COL_SLICE  = slice(6, 11)

# -----------------------------
# 2. TOP-K VIA BORDA (AVG RANK)
# -----------------------------
def get_top_features_from_csv_borda(csv_path, top_k):
    """
    BORDA / Average-rank aggregation:
      - For each metric column, rank features by descending score (rank 1 = best)
      - Average ranks across the metric columns
      - Sort by smallest average rank
    """
    df = pd.read_csv(csv_path)

    feature_names = df.iloc[:, 0].astype(str)

    bhc_metrics = df.iloc[:, BHC_COL_SLICE].copy()
    di_metrics  = df.iloc[:, DI_COL_SLICE].copy()

    # Rank each metric column: higher score => better rank (1 is best)
    bhc_ranks = bhc_metrics.rank(ascending=False, method="average")
    di_ranks  = di_metrics.rank(ascending=False, method="average")

    # Borda = average rank
    bhc_borda = bhc_ranks.mean(axis=1)
    di_borda  = di_ranks.mean(axis=1)

    df_bhc = pd.DataFrame({"feature": feature_names, "borda": bhc_borda})
    df_di  = pd.DataFrame({"feature": feature_names, "borda": di_borda})

    top_bhc = df_bhc.sort_values("borda", ascending=True).head(top_k)["feature"].tolist()
    top_di  = df_di.sort_values("borda",  ascending=True).head(top_k)["feature"].tolist()

    return top_bhc, top_di

# -----------------------------
# 3. FEATURE ABLATION UTILITY
# -----------------------------
def create_leave_one_out_sets(feature_list):
    """Returns list of (removed_feature, remaining_feature_list)."""
    return [(f, [x for x in feature_list if x != f]) for f in feature_list]

# -----------------------------
# 4. INPUT FORMATTING
# -----------------------------
ALL_POSSIBLE_FEATURES = [
    "Sex", "Department_Service", "Chief_Complaint",
    "History_Present_Illness", "Past_Medical_Surgical_History",
    "Family_History", "Personal_Social_History",
    "Drug_History_Allergies", "Clinical_Examination",
    "Investigations", "Treatment_During_Stay",
    "Procedures_Surgeries_Done", "Final_Diagnosis",
    "Condition_at_Discharge", "Discharge_Medications",
    "Discharge_Disposition", "Followup_Instructions"
]

def format_data(d, keep_only_list):
    keep_set = set(keep_only_list)
    lines = []
    for key in ALL_POSSIBLE_FEATURES:
        if key in keep_set:
            val = d.get(key, "")
            if val and str(val) != "nan":
                lines.append(f"{key}:\n{val}\n")
    return "\n".join(lines)

# -----------------------------
# 5. LOAD MODEL WITH ADAPTERS
# -----------------------------
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": DEVICE_INDEX},
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, BHC_ADAPTER_PATH, adapter_name="bhc")
    model.load_adapter(DI_ADAPTER_PATH, adapter_name="di")
    model.eval()

    return model, tokenizer

# -----------------------------
# 6. GENERATION FUNCTION
# -----------------------------
@torch.no_grad()
def generate(model, tokenizer, text_input, task, max_new_tokens=512):
    model.set_adapter(task)

    sys_msg = (
        "You are a medical AI. Write the Hospital Course."
        if task == "bhc"
        else "You are a medical AI. Write Discharge Advice."
    )
    header = "Hospital Course" if task == "bhc" else "Discharge Advice"

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Analyze the following clinical data:\n{text_input}\n\nTask: Write {header}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{DEVICE_INDEX}")

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# -----------------------------
# 7. MAIN EXECUTION (PRUNING + LOFO ONLY)
# -----------------------------
if __name__ == "__main__":

    # 7.1 Pick Top-K via BORDA
    top_bhc_features, top_di_features = get_top_features_from_csv_borda(CSV_PATH, TOP_K)
    print("Top BHC features (BORDA):", top_bhc_features)
    print("Top DI features  (BORDA):", top_di_features)

    # 7.2 Load model
    model, tokenizer = load_model()

    # 7.3 Read input data
    with open(INPUT_JSONL) as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    subset_data = all_data[:SAMPLE_SIZE]

    # 7.4 Run baseline + leave-one-out ablations and save outputs
    with open(FINAL_OUTPUT, "w") as fout:
        for pid, patient in enumerate(tqdm(subset_data, desc="LOFO")):

            # -------- BHC BASELINE --------
            bhc_full_input = format_data(patient, top_bhc_features)
            bhc_baseline = generate(model, tokenizer, bhc_full_input, "bhc")

            bhc_ablations = []
            for removed, feat_set in create_leave_one_out_sets(top_bhc_features):
                ablated_input = format_data(patient, feat_set)
                pred = generate(model, tokenizer, ablated_input, "bhc")
                bhc_ablations.append({
                    "removed_feature": removed,
                    "features_used": feat_set,
                    "prediction": pred
                })

            # -------- DI BASELINE --------
            di_full_input = format_data(patient, top_di_features)
            di_baseline = generate(model, tokenizer, di_full_input, "di")

            di_ablations = []
            for removed, feat_set in create_leave_one_out_sets(top_di_features):
                ablated_input = format_data(patient, feat_set)
                pred = generate(model, tokenizer, ablated_input, "di")
                di_ablations.append({
                    "removed_feature": removed,
                    "features_used": feat_set,
                    "prediction": pred
                })

            # -------- Write record --------
            record = {
                "patient_id": pid,

                "bhc_baseline": {
                    "features_used": top_bhc_features,
                    "prediction": bhc_baseline
                },
                "bhc_leave_one_out": bhc_ablations,

                "di_baseline": {
                    "features_used": top_di_features,
                    "prediction": di_baseline
                },
                "di_leave_one_out": di_ablations,

                "bhc_ground_truth": patient.get("Hospital_Course", "N/A"),
                "di_ground_truth": patient.get("Discharge_Advice", "N/A")
            }

            fout.write(json.dumps(record) + "\n")

    print(f"✅ Pruning inference (BORDA top-{TOP_K} + LOFO) saved to: {FINAL_OUTPUT}")
