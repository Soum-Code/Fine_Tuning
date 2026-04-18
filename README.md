# Atom of Thoughts (AoT) Fine-Tuning System

Industrial-grade fine-tuning system implementing the **Atom of Thoughts** (AoT) reasoning framework. Designed to run on limited-resource devices using PEFT and 4-bit quantization.

---

## What this project does

Standard Chain-of-Thought (CoT) prompting is linear and memory-heavy. AoT treats reasoning as a **Markovian process**:

1. **Decomposition** — breaks a complex problem into independent atomic states
2. **Atomic Reasoning** — solves each state in isolation, preventing history interference
3. **Contraction** — merges atomic solutions into a final, verifiable answer

This reduces token bloat and improves reasoning accuracy on complex tasks.

---

## System architecture

- `src/training/aot_engine.py` — Core Decompose-Solve-Contract loop
- `src/model/model_manager.py` — Model loading with MXFP4 and NF4 4-bit quantization
- `src/training/trainer.py` — Unified training pipeline for multi-scale models (0.5B to 20B)
- `local_lite/` — Optimized sub-system for CPU-only and 16GB RAM environments

---

## Getting started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training with AoT

```bash
python src/training/trainer.py --model qwen_7b --template aot --dataset ./data/aot_research_data.json
```

Set `WANDB_MODE=disabled` if you are not using Weights and Biases:

```bash
WANDB_MODE=disabled python src/training/trainer.py --model qwen_7b --template aot --dataset ./data/aot_research_data.json
```

### Sync changes to Git

To push local changes to the repository, run the sync script from the project root:

```powershell
./scripts/sync_to_git.ps1
```

This script stages all modified files, commits with a timestamp message, and pushes to the main branch.

---

## Current status

- Smoke tests confirmed successful inference on Qwen2.5-0.5B with 4-bit NF4 quantization on CPU
- Training loop runs end-to-end with AoT decompose-solve-contract prompt structure
- Full evaluation on a held-out reasoning benchmark is in progress

---

## Tech stack

Python · PyTorch · Transformers · PEFT · BitsAndBytes · Qwen2.5 · WandB

---

## License

MIT
