# 📑 Project Title: BioGemma-Guard

### *Decentralized Verifiable Reasoning for Clinical Triage & Hallucination Mitigation*

---

## 🎯 Project Overview

**BioGemma-Guard** adalah sistem asisten triase medis berbasis **Gemma 4** yang dirancang untuk lingkungan kritis (UGD). Proyek ini mengintegrasikan tiga pilar utama AI modern:  **Reasoning (GRPO)** ,  **Statistical Safety (Conformal Prediction)** , dan  **Verifiable Execution (Sandbox)** .

Tujuan utamanya adalah menciptakan AI medis yang "Jujur": ia memberikan diagnosa yang akurat secara logika, atau memilih untuk **Abstain** (diam) jika tingkat ketidakpastian terlalu tinggi, sehingga mencegah malpraktik akibat halusinasi AI.

---

## 🛠️ Technical Stack (The "Engineering" Meat)

### 1. Training & Optimization

* **Model:** Google Gemma 4 (Fine-tuned via  **Unsloth** ).
* **Algorithm:**  **GRPO (Group Relative Policy Optimization)** . Menghilangkan model *Critic* untuk efisiensi VRAM hingga 50%, memungkinkan training model *reasoning* pada  *distributed infrastructure* .
* **Memory Ops:** FlashAttention-2, ZeRO-3 Sharding, dan 4-bit QLoRA untuk memungkinkan  *high-throughput training* .

### 2. Safety & Verification Layer

* **Reasoning Engine:** Menggunakan *Chain-of-Thought* (CoT) untuk transparansi langkah diagnosa.
* **Statistical Gate:** **Conformal Prediction** (CP) untuk memberikan jaminan matematis pada error rate (misal: **$\alpha = 0.05$**).
* **Execution Sandbox:** Menggunakan Python `subprocess` + Docker untuk memverifikasi kalkulasi dosis medis dan skor klinis (GCS, NEWS2) secara terisolasi.

---

## 📂 Repository Structure (Suggested)

**Bash**

```
BioGemma-Guard/
├── data/
│   ├── raw/                # Dataset MedQA / PubMedQA
│   └── processed/          # GRPO-ready triplets (Prompt, Reward, Constraint)
├── src/
│   ├── training/
│   │   ├── grpo_trainer.py # Custom TRL implementation for Gemma 4
│   │   └── unsloth_config.py
│   ├── inference/
│   │   ├── conformal_calibrator.py # CP logic for abstention
│   │   └── sandbox_env.py  # Secure code execution logic
│   └── rewards/
│       └── medical_reward.py # Reward functions (Accuracy + Honesty)
├── notebooks/
│   └── 01_benchmark_reasoning.ipynb
├── docker/
│   └── sandbox.Dockerfile  # Isolated env for medical code execution
└── README.md
```


## 🚀 Key Features for the Pitch

### 1. "Hallucination-Proof" Triage

Menggunakan  **RLVR (Reinforcement Learning with Verifiable Rewards)** . Model tidak hanya dinilai dari kemiripan teks, tapi dari kebenaran logika medis yang bisa diverifikasi oleh sistem (misal: apakah hitungan dosisnya benar di Sandbox?).

### 2. Statistical Trust with Conformal Prediction

Tidak seperti LLM biasa yang hanya memberikan "probabilitas menebak", BioGemma-Guard memberikan set prediksi dengan jaminan cakupan. Jika set tersebut terlalu luas, sistem memicu  **Abstention** .

### 3. Edge-Ready Deployment

Optimasi menggunakan **LiteRT** atau  **Ollama** , sehingga sistem asisten triase ini bisa berjalan secara *offline* di puskesmas atau daerah bencana (relevan untuk kategori  *Global Resilience* ).

---

## 📈 Impact Analysis (For Hackathon)

* **Safety:** Menurunkan risiko *Under-triage* pada kasus kritis (seperti serangan jantung atipikal).
* **Efficiency:** Mengurangi beban kerja dokter dengan melakukan pra-analisis data pasien secara otomatis.
* **Trust:** Membangun jembatan antara teknologi AI dan kebijakan medis melalui transparansi penalaran.

### Langkah Selanjutnya:

1. **Environment Setup:** Instal `unsloth`, `trl`, dan `transformers` di Kaggle/Colab.
2. **Base Model:** Load Gemma 4 (4-bit).
3. **Initial Calibration:** Gunakan dataset medis kecil untuk menghitung *p-value* awal menggunakan **Conformal Prediction** yang sudah kamu punya.
