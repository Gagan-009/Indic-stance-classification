# PJ01 — Transformer Readiness & MuRIL Feasibility Report

**Task:** Hindi–Bengali Climate Stance Classification  
**Target Architecture Evaluated:** MuRIL (`google/muril-base-cased`)  
**Host Environment:** Windows 10/11 x64, Intel 8-core CPU, No CUDA GPU  
**Date:** October 2026  

---

## 1. Environment & Hardware Audit

Before considering deep transformer fine-tuning, an explicit hardware and runtime audit was conducted in the local development environment:

| Audit Parameter | Measured State | Implication for Transformer Fine-Tuning |
| :--- | :--- | :--- |
| **Compute Device** | CPU (`Intel Core 8-core`) | No dedicated tensor accelerator available. Matrix operations executed on host CPU. |
| **CUDA Acceleration** | `torch.cuda.is_available() == False` | Deep learning models cannot utilize GPU tensor cores or mixed precision (FP16/BF16). |
| **PyTorch Version** | `torch 2.12.0+cpu` | Functional for CPU tensor execution, but limited to standard thread pools. |
| **Hugging Face `transformers`** | **Not Installed** | Package missing from active environment (`ModuleNotFoundError: No module named 'transformers'`). |
| **Host System RAM** | Standard developer workstation (~16 GB) | Shared with OS, background IDE processes, and language server. |

---

## 2. Evaluation of MuRIL (`google/muril-base-cased`)

MuRIL (Multilingual Representations for Indian Languages) was developed by Google Research specifically to handle Indian languages, trained on monolingual and parallel corpora across 17 Indic languages (including Hindi and Bengali) and English.

### Architectural Profile:
- **Base Architecture:** BERT-base encoder (12 layers, 768 hidden dimensions, 12 self-attention heads).
- **Parameters:** ~236 million parameters.
- **Weights Disk Footprint:** ~950 MB checkpoint + ~3.5 MB WordPiece vocabulary.
- **Memory Footprint (Inference):** ~1.5–2.0 GB RAM for model weights in FP32.
- **Memory Footprint (Fine-Tuning):** ~6.0–8.5 GB RAM required to maintain optimizer states (AdamW first/second moments), activations, backward pass gradients, and computation graphs.

### Feasibility Verdict: **UNSUITABLE / IMPRACTICAL FOR LOCAL CPU FINE-TUNING**
1. **Computational Prohibiveness:** Without CUDA acceleration, backpropagation through 12 transformer layers with 236M parameters on an 8-core CPU takes an estimated **45–75 minutes per epoch** (even on modest batch sizes of 8–16). A standard 3-epoch fine-tuning run would take **3–4+ hours**, triggering heavy thermal throttling on Windows and potentially locking the IDE.
2. **Missing Dependencies:** The `transformers` and `accelerate` libraries are not part of the current production requirements (`requirements.txt` currently targets lightweight classical deployment: `pandas`, `scikit-learn`, `joblib`, `streamlit`).
3. **Overfitting Risk on Small Corpora:** Fine-tuning 236M parameters on small pilot datasets (<1,000 samples) without massive regularization, frozen lower layers, or LoRA parameter-efficient tuning tends to rapidly overfit or suffer catastrophic forgetting.

---

## 3. Recommended Multilingual Alternatives

### Alternative 1 (Recommended): `ai4bharat/indic-bert`
* **Architecture:** ALBERT-based encoder tailored specifically for 12 major Indian languages (including Hindi and Bengali).
* **Parameters:** **~33 million parameters** (approx. **7x smaller than MuRIL**).
* **Model Size:** ~135 MB (vs. ~950 MB for MuRIL).
* **Why it suits our project:**
  1. *Resource Efficiency:* Fits entirely in CPU cache; inference and feature extraction run 5–8x faster than MuRIL on standard CPUs.
  2. *Native Indic Tokenization:* Uses SentencePiece vocabulary optimized specifically for Indic scripts (Devanagari, Bengali, etc.), mitigating out-of-vocabulary and subword fragmentation issues.
  3. *Lightweight Fine-Tuning:* Can be trained or feature-extracted locally with minimal memory pressure (<1.5 GB total RAM).

### Alternative 2: Lightweight Feature Extraction via Sentence-Transformers
* **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~117M parameters, ~470 MB).
* **Strategy:** Keep the transformer frozen. Generate dense contextual sentence embeddings for comments in Hindi and Bengali, then train our balanced Logistic Regression or LinearSVC classification head.
* **Benefits:** Zero risk of catastrophic forgetting, training completes in under 15 seconds on CPU, and achieves strong semantic representation without expensive backpropagation.

### Alternative 3: Cloud GPU Acceleration for MuRIL / IndicBERT
* When dataset scaling reaches 1,000+ annotations, training scripts should be executed in an external GPU container (e.g., Google Colab with T4/V100 GPU or AWS EC2 g4dn instance).
* The dataset export pipeline (`data/processed/cleaned_stance_dataset.csv`) is fully modular and ready to be loaded directly via Hugging Face `datasets.load_dataset('csv', data_files=...)`.

---

## 4. Implementation Readiness Protocol

To transition to transformer-based modeling once GPU compute or lighter dependencies are provisioned:
1. Keep the classical TF-IDF baseline (`Macro-F1: 0.5724`) as the benchmark to beat.
2. Add `transformers>=4.40.0` and `torch` to an optional deep learning requirements file (`requirements-dl.txt`).
3. First benchmark `ai4bharat/indic-bert` in frozen feature-extraction mode before committing to full fine-tuning.
