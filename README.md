# ⚛️ Atom of Thoughts (AoT) Fine-Tuning System

Industrial-grade fine-tuning system designed to implement and scale the **Atom of Thoughts** (AoT) reasoning framework on any device, even with limited resources.

## 🚀 Project Motive
The goal of this project is to democratize high-level reasoning research. By leveraging **Parameter-Efficient Fine-Tuning (PEFT)** and **4-bit Quantization**, we enable researchers to train large models (up to 20B+) on consumer-grade hardware or CPU-only environments.

## 🧠 What is Atom of Thoughts (AoT)?
Unlike standard Chain-of-Thought (CoT) which is linear and memory-heavy, AoT treats reasoning as a **Markovian Process**:
1.  **Decomposition**: Breaking complex problems into independent 'atomic' states.
2.  **Atomic Reasoning**: Solving each state in isolation to prevent history-interference.
3.  **Contraction**: Merging atomic solutions into a final, verifiable answer.

This method drastically reduces token bloat and improves reasoning accuracy for complex mathematical and research tasks.

## 🏗️ System Architecture
- **`src/training/aot_engine.py`**: The core orchestrator for the Decompose-Solve-Contract loop.
- **`src/model/model_manager.py`**: Handles industrial model loading with native support for MXFP4 and NF4 quantization.
- **`src/training/trainer.py`**: Unified training pipeline supporting multi-scale models (0.5B, 7B, 20B).
- **`local_lite/`**: Optimized sub-system for ultra-low resource (CPU/16GB RAM) training.

## 🛠️ Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training AoT (Research Batch)
```powershell
# Optimized for CPU/Limited Resource
$env:WANDB_MODE="disabled"; python src/training/trainer.py --model qwen_7b --template aot --dataset ./data/aot_research_data.json
```

### 🔄 Auto-Sync Protocol
To ensure the repository is always up-to-date, I have implemented an auto-sync utility. You can run it anytime you make changes:
```powershell
./scripts/sync_to_git.ps1
```
*Note: As your AI assistant, I will automatically run this sync after major upgrades.*

## 📈 Current Status
- ✅ **Infrastructure Verified**: Smoke tests successful on Qwen2.5-0.5B.
- ✅ **Quantization Hardened**: Native support for pre-quantized 20B models confirmed.
- ✅ **Auto-Sync Active**: GitHub repository is linked and automated.

---
*Developed for advanced AI research in reasoning and scaling.*