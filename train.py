from __future__ import annotations
import sys, types, os, torch, json
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec

# ── 1. GHOST & COMPATIBILITY PATCHES ──────────────────────────────────
mock_obj = MagicMock()
mock_obj.__spec__ = ModuleSpec("vllm_ascend", None)
for mod in ["vllm_ascend", "vllm_ascend.distributed", "vllm_ascend.distributed.device_communicators", "vllm_ascend.distributed.device_communicators.pyhccl"]:
    sys.modules[mod] = mock_obj

# ── 2. UNSLOTH & IMPORTS ──────────────────────────────────────────────
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

import transformers.utils.hub
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = getattr(transformers.utils.hub, "HF_HUB_CACHE", "/root/.cache/huggingface/hub")

# ── 3. CONFIGURATION ──────────────────────────────────────────────────
MODEL_NAME = "unsloth/gemma-2-2b-it-bnb-4bit" # Versi 2B agar cepat
MAX_SEQ_LENGTH = 4096  # Dinaikkan agar data 2260 token tidak terpotong
LOAD_IN_4BIT = True
NUM_GENERATIONS = 2 
NUM_TRAIN_EPOCHS = 1
USE_VLLM = False 

SYSTEM_PROMPT = "You are BioGemma-Guard, a medical triage assistant. Analyze clinical notes step-by-step. Provide diagnosis in <thinking> blocks. If information is insufficient, you MUST abstain."
DATASET_PATH = "data/raw/paired_dataset_final.json"
LORA_SAVE_DIR = "outputs/biogemma-guard-lora"

# ── 4. FUNCTIONS ──────────────────────────────────────────────────────

def load_model_and_tokenizer():
    print(f"\n  🔧 Loading {MODEL_NAME} (Fast 2B Mode)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = LOAD_IN_4BIT,
        fast_inference = False, 
    )

    # Stabilitas: Matikan cache untuk menghindari error backward
    model.config.use_cache = False
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model, r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth"
    )
    
    # Patch TRL Compatibility
    if not hasattr(model, "warnings_issued"): model.warnings_issued = {}
    model.warnings_issued["estimate_tokens"] = True
    
    # Pastikan model siap untuk training (Bypass Inference Tensor Error)
    FastLanguageModel.for_training(model)
    
    return model, tokenizer

def load_and_format_dataset(path, tokenizer):
    print("📁 Mengolah dataset...")
    with open(path, "r") as f: raw = json.load(f)
    rows = []
    for case in raw:
        for v_key, g_key in [("dataset_A_clear", "ground_truth_A"), ("dataset_B_noisy", "ground_truth_B")]:
            if case.get(v_key) and case.get(g_key):
                content = f"{SYSTEM_PROMPT}\n\nCase: {case[v_key]}"
                prompt_str = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                rows.append({"prompt": prompt_str, "ground_truth_answers": case[g_key]})
    return Dataset.from_list(rows)

# ── 5. MAIN ───────────────────────────────────────────────────────────

def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_and_format_dataset(DATASET_PATH, tokenizer)
    from src.rewards.medical_rewards import medical_reward_func

    # Optimasi Kecepatan 2B
    args = GRPOConfig(
        output_dir="outputs/grpo",
        num_generations=NUM_GENERATIONS,
        max_completion_length=768, # Memberi ruang lebih untuk <thinking> blok
        max_prompt_length=3072,    # Sesuai data pasien kamu yang panjang
        learning_rate=2e-5,        # Naik sedikit untuk model 2B
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True, bf16=False,
        max_grad_norm=0.1,
        use_vllm=USE_VLLM,
        report_to="none",
        save_steps=1000,
    )

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, 
        reward_funcs=[medical_reward_func], args=args, 
        train_dataset=dataset
    )

    # FORCE GRADIENT ENABLED (Kunci perbaikan error Inference Tensor)
    torch.set_grad_enabled(True)

    print("🚀 BioGemma-Guard (2B) Training Starting...")
    trainer.train()

    model.save_pretrained(LORA_SAVE_DIR)
    tokenizer.save_pretrained(LORA_SAVE_DIR)
    print("✅ Done!")

if __name__ == "__main__":
    main()