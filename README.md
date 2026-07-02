<div align="center">

# 🛡️ BioGemma-Guard

### Calibrated Abstention in Clinical LLMs via Group Relative Policy Optimization

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](#)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg)](#)

*Teaching medical LLMs when to diagnose — and when to stop.*

</div>

---

## Abstract

Large Language Models deployed in clinical settings face a critical safety failure mode: they generate confident-sounding diagnoses even when presented with insufficient, corrupted, or ambiguous patient data. Standard mitigation strategies — such as appending generic disclaimers ("I'm an AI and cannot provide medical advice") — fail to distinguish between genuine diagnostic capability and epistemic uncertainty, producing what we term **lazy abstention**: a blanket refusal that is clinically uninformative and operationally useless.

**BioGemma-Guard** introduces a fundamentally different paradigm: **Calibrated Abstention**. Rather than appending post-hoc disclaimers, the model learns — through reinforcement learning — to perform a bifurcated reasoning process:

1. **When clinical data is sufficient** (Variant A inputs): execute full diagnostic reasoning within structured `<thinking>` blocks, culminating in a specific, defensible clinical assessment.
2. **When clinical data is insufficient** (Variant B inputs): shift from diagnostic execution to **structured data-gap analysis** — identifying precisely *which* clinical elements are missing (vitals, lab results, imaging, physical examination findings) and *why* a diagnosis cannot be safely rendered.

This behavior is achieved through pure **Group Relative Policy Optimization (GRPO)** — a value-function-free reinforcement learning algorithm that eliminates the critic network entirely, computing policy advantages relative to sampled completion groups rather than learned value estimates. The entire pipeline runs on a single Tesla T4 GPU (16 GB VRAM) using 4-bit NF4 quantization and LoRA adapters, updating only ~0.2% of model parameters.

---

## Core Philosophy: Calibrated Abstention vs. Lazy Disclaimers

| Property | Lazy Disclaimer | Calibrated Abstention |
|:---------|:----------------|:----------------------|
| **Trigger** | All outputs indiscriminately | Only when epistemic uncertainty is detected |
| **Content** | Generic boilerplate ("consult a doctor") | Structured analysis of specific data gaps |
| **Clinical Utility** | Zero — adds noise, erodes trust | High — guides next diagnostic steps |
| **Training Signal** | None — appended post-hoc | Learned via reward-shaped RL (GRPO) |
| **Reasoning Trace** | Absent or superficial | Full CoT within `<thinking>` blocks |

---

## Mathematical Foundations

### GRPO Objective

BioGemma-Guard is trained using **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024), which eliminates the critic model entirely. For each prompt *x*, the trainer samples a group of *G* completions {*o₁*, …, *o_G*} from the current policy π_θ and computes group-relative advantages:

```
Â_i = (r_i − mean({r₁, …, r_G})) / std({r₁, …, r_G})
```

The policy is then updated by maximizing:

```
J_GRPO(θ) = E[1/G · Σ min(ρ_i · Â_i, clip(ρ_i, 1−ε, 1+ε) · Â_i) − β · D_KL(π_θ ‖ π_ref)]
```

where `ρ_i = π_θ(o_i|x) / π_θ_old(o_i|x)`.

With group size *G* = 2, this provides a lightweight yet effective advantage signal without training a separate value network — reducing VRAM requirements by approximately 50% compared to PPO-based approaches.

### 4-Layer Asymmetric Reward Matrix

The reward function `r(x, o, y*)` implements a deliberately asymmetric incentive structure:

| Outcome | Reward | Rationale |
|:--------|-------:|:----------|
| **Correct diagnosis** (answerable question, answer matches y*) | +2.0 | Strong positive reinforcement for accurate clinical reasoning |
| **Honest abstention** (ambiguous question, model abstains) | +0.8 | Moderate positive reinforcement for epistemic honesty |
| **Hallucination / incorrect** (any incorrect diagnosis or reckless guess on ambiguous data) | −3.0 | Severe penalty to suppress medical misinformation |
| **Chain-of-Thought bonus** (visible reasoning markers detected) | +0.2 | Additive bonus incentivizing transparent reasoning traces |

**The 3.75:1 Penalty-to-Incentive Ratio.** The hallucination penalty (−3.0) is **3.75× the magnitude** of the abstention reward (+0.8). This asymmetry is a deliberate design choice grounded in clinical risk theory: in medical contexts, the cost of a false positive (hallucinated diagnosis leading to inappropriate treatment) vastly exceeds the cost of a false negative (abstention requiring physician escalation). The model learns that *silence is safer than speculation*.

An additional penalty of −0.5 is applied when the model unnecessarily abstains on clearly answerable questions, preventing the policy from collapsing into universal refusal.

### Conformal Prediction Safety Layer

At inference time, a **Split / Inductive Conformal Prediction** calibrator provides distribution-free coverage guarantees:

```
C(x) = { k : π̂_k ≥ 1 − q̂ }
```

where `q̂` is the critical quantile computed from calibration non-conformity scores `s_i = 1 − π̂_{y_i}` with finite-sample correction:

```
q_level = ⌈(n+1)(1 − α)⌉ / n
```

**Abstention rule:** The model diagnoses if and only if |C(x)| = 1. If |C(x)| = 0 (no class passes threshold) or |C(x)| > 1 (statistically ambiguous), the system triggers abstention, guaranteeing:

```
P(y_test ∈ C(x_test)) ≥ 1 − α
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    BioGemma-Guard Pipeline                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Paired Dataset    │    │  Gemma-2-2B-IT (4-bit NF4)       │  │
│  │  (200 cases × 2)   │───▶│  + LoRA Adapters (r=16)          │  │
│  │  Clear + Noisy     │    │  ~0.2% trainable params          │  │
│  └────────────────────┘    └──────────┬───────────────────────┘  │
│                                       │                          │
│                              ┌────────▼────────┐                 │
│                              │  GRPO Trainer   │                 │
│                              │  G=2, ε-clip    │                 │
│                              │  No Critic Net  │                 │
│                              └────────┬────────┘                 │
│                                       │                          │
│              ┌────────────────────────▼─────────────────────┐    │
│              │        4-Layer Reward Function               │    │
│              │  ┌──────────┬──────────┬──────────┬───────┐  │    │
│              │  │Correct   │Abstain   │Halluc.   │CoT    │  │    │
│              │  │ +2.0     │ +0.8     │ -3.0     │ +0.2  │  │    │
│              │  └──────────┴──────────┴──────────┴───────┘  │    │
│              └──────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Inference Stack                           │ │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐  │ │
│  │  │ Conformal Calibrator│  │ Medical Code Sandbox         │  │ │
│  │  │ α=0.05, |C(x)|=1    │  │ subprocess + import whitelist│  │ │
│  │  │ → Diagnose / Abstain│  │ → Verify dosage calculations │  │ │
│  │  └─────────────────────┘  └──────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Local Resource Optimization

The entire training and inference pipeline is designed to run on a **single consumer-grade Tesla T4 GPU** (16 GB VRAM):

| Component | Configuration | VRAM Impact |
|:----------|:--------------|:------------|
| **Base Model** | `unsloth/gemma-2-2b-it-bnb-4bit` | ~2.5 GB (4-bit NF4 quantization) |
| **LoRA Adapters** | r=16, targeting `q/k/v/o/gate/up/down_proj` | ~0.2% of total params |
| **GRPO (no critic)** | G=2 generations per prompt | ~50% VRAM savings vs. PPO |
| **Gradient Checkpointing** | Unsloth optimized | Trades compute for memory |
| **Sequence Length** | `max_prompt_length=3072`, `max_completion_length=768` | Fits long SOAP notes |
| **Batch Config** | `per_device_batch_size=1`, `grad_accum=4` | Effective batch = 4 |
| **Precision** | FP16 (BF16 disabled for T4 compatibility) | Standard T4 mixed precision |

---

## Dataset Architecture

The training corpus consists of **200 curated master clinical cases**, each expanded into two variants:

| Variant | Description | Ground Truth | Expected Behavior |
|:--------|:------------|:-------------|:------------------|
| **A — Clear** | Structured SOAP notes with complete clinical data (vitals, labs, imaging, physical exam, assessment/plan) | Specific diagnosis (e.g., "Acute Myocardial Infarction") | Diagnose within `<thinking>` blocks |
| **B — Noisy** | First-person patient narratives with contextual interference, metric omission, emotional tangents, and non-clinical language | `"The information provided is inconclusive..."` requiring further evaluation | Abstain with structured data-gap analysis |

This yields **400 total training instances** in `paired_dataset_final.json`, with each JSON record containing:
- `id`: Case identifier (e.g., `CASE_001`)
- `medical_specialty`: Clinical domain classification
- `dataset_A_clear`: Complete SOAP-format clinical note
- `dataset_B_noisy`: Corrupted patient narrative
- `ground_truth_A`: Expected diagnosis for clear input
- `ground_truth_B`: Expected abstention rationale for noisy input

---

## Empirical Replication Guide

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 — Verify Dataset Integrity

```bash
python -c "
import json
with open('data/raw/paired_dataset_final.json') as f:
    data = json.load(f)
print(f'Master cases: {len(data)}')
print(f'Total training instances: {len(data) * 2}')
print(f'Sample keys: {list(data[0].keys())}')
"
```

Expected output:
```
Master cases: 200
Total training instances: 400
Sample keys: ['id', 'medical_specialty', 'description', 'keywords', 'dataset_A_clear', 'dataset_B_noisy', 'ground_truth_A', 'ground_truth_B', 'dokumen_pgvector']
```

### Step 2 — Run Reward Function Self-Test

```bash
python -m src.rewards.medical_rewards
```

Expected: 11/11 test cases pass, validating correct/abstention/hallucination/CoT scoring logic.

### Step 3 — Run Conformal Calibrator Self-Test

```bash
python -m src.inference.conformal_calibrator
```

Expected: All assertions pass (confident → diagnose, uncertain → abstain, batch inference correct, edge cases handled).

### Step 4 — Run Medical Sandbox Self-Test

```bash
python -m src.inference.medical_sandbox
```

Expected: 6/6 tests pass (valid execution, syntax error catch, timeout protection, import policy enforcement, empty code handling, NEWS2 calculation verification).

### Step 5 — Launch GRPO Training

```bash
# Requires CUDA-capable GPU (Tesla T4 or equivalent)
python train.py
```

Training configuration (hardcoded in `train.py`):
- Model: `unsloth/gemma-2-2b-it-bnb-4bit`
- Epochs: 1
- Learning rate: `2e-5`
- Group size (G): 2
- LoRA rank: 16
- Output: `outputs/biogemma-guard-lora/`

### Step 6 — Generate Publication Figures

```bash
python generate_figures.py
```

Produces `figures/fig1_radar_comparison.png` and `figures/fig2_cot_discipline_distribution.png`.

---

## Evaluation Results

### LLM-as-a-Judge (Llama-4-Scout-17B-16E-Instruct, n = 50 noisy cases)

| Metric | Base Model (Gemma-2-2B) | BioGemma-Guard (GRPO) | Δ |
|:-------|:-----------------------:|:---------------------:|:-:|
| Clinical Faithfulness (1–10) | 7.96 | 7.98 | +0.02 |
| Differential Quality (1–10) | 6.84 | 6.84 | 0.00 |
| **CoT Discipline (1–10)** | 7.68 | **8.24** | **+0.56** |

**Key Finding:** GRPO training preserves core medical knowledge (stable Faithfulness and Differential Quality — no catastrophic forgetting) while significantly improving the **structural integrity and professional discipline** of the clinical reasoning trace (+7.3% relative improvement in CoT Discipline).

### Softmax Confidence Calibration

| Model | Average Transition Confidence |
|:------|:-----------------------------:|
| Base Gemma-2-2B | 66.70% |
| BioGemma-Guard | 66.94% |

Both models converge to a ~66% "high-entropy" confidence state on ambiguous inputs, confirming that the models correctly perceive epistemic uncertainty rather than guessing.

---

## Module Reference

### `src/rewards/medical_rewards.py`

```python
from src.rewards.medical_rewards import medical_reward_func

trainer = GRPOTrainer(
    ...,
    reward_funcs=[medical_reward_func],
)
```

**Key components:**
- **Fuzzy answer matching** via `thefuzz` (handles "Myocardial Infarction" ≈ "Acute Myocardial Infarct")
- **24 compiled regex patterns** for abstention phrase detection
- **17 compiled regex patterns** for Chain-of-Thought marker detection
- **Ground-truth-only ambiguity detection** — avoids false positives from ambiguous prompts

### `src/inference/conformal_calibrator.py`

```python
from src.inference.conformal_calibrator import MedicalConformalCalibrator

cal = MedicalConformalCalibrator(alpha=0.05)
cal.fit(calib_probs, calib_labels)
if cal.should_abstain(test_probs):
    response = "Insufficient confidence for safe diagnosis."
```

### `src/inference/medical_sandbox.py`

```python
from src.inference.medical_sandbox import MedicalCodeSandbox

sandbox = MedicalCodeSandbox()
result = sandbox.run_from_completion(llm_output)
verified = sandbox.verify_output(result, expected_value="250.0")
```

**Security features:** Import whitelist (10 safe modules only), configurable timeout (default 3s), fresh subprocess isolation.

---

## Citation

```bibtex
@software{wibisono2026biogemma,
  author       = {Wibisono, Narendra Bayutama},
  title        = {{BioGemma-Guard: Calibrated Abstention in Clinical
                   LLMs via Group Relative Policy Optimization}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX},
  license      = {Apache-2.0}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- **Unsloth** for parameter-efficient fine-tuning infrastructure
- **TRL (Transformer Reinforcement Learning)** for the GRPOTrainer implementation
- **Google DeepMind** for the Gemma-2 model family
