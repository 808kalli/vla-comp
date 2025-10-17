# Vision-Language-Action Models are Efficient Learners

---

## 📖 Overview

This repository contains the code and experimental setup for the project **"Vision-Language-Action (VLA) Models are Efficient Learners"**.

This project benchmarks **OpenVLA (7B)** and **SmolVLA (2B)** on the **LIBERO-Spatial** simulation benchmark using a 7-DoF Franka Emika Panda robot. I explore how efficiently these models adapt to new robotic environments, how model size affects generalization, and what energy–performance trade-offs emerge during fine-tuning.

The focus is on:

1. How efficiently can VLA models adapt to **new embodiments, camera views, and environments**?
2. How does **model scale** (VLM backbone capacity) affect zero-shot and transfer learning?
3. What are the **energy–performance trade-offs** between large and compact architectures?

---

## ⚙️ Experimental Setup

### Simulation Environment
- **Benchmark:** LIBERO-Spatial (10 spatial reasoning tasks)
- **Robot:** Franka Emika Panda, 7-DoF arm  
- **Dataset:** 434 expert demonstrations (RLDS format via HuggingFace)

### Models Evaluated
| Model | Parameters | Backbone | Vision Encoder |
|:------|:------------|:----------|:----------------|
| **OpenVLA** | 7B | LLaMA-2 | DINOv2 + SigLIP |
| **SmolVLA** | 2B | Phi-3 | SigLIP |

---

## 🧪 Experiments

| Regime | Description |
|:--------|:-------------|
| **Zero-Shot** | Direct evaluation without fine-tuning |
| **Few-Shot (10 / 100)** | Adaptation with limited expert demonstrations |
| **Full Fine-Tuning** | Full dataset training |
| **Scratch (SmolVLA only)** | Train from scratch keeping vision & language frozen |

Both models were fine-tuned in simulated LIBERO environments, with **LoRA adapters** for efficient training.

### 🎯 Task Completion Visualization
<img width="709" height="338" alt="task_completion" src="https://github.com/user-attachments/assets/19d93381-c216-43b6-abb9-8667ee40888e" />

*Figure 1: Example rollout from the LIBERO-Spatial benchmark — the robot successfully grasps and places the black bowl according to the language instruction.*

---

## 📊 Results Summary

| Model | Zero-Shot | 10 Ep | 100 Ep | Full | Scratch |
|:------|:-----------|:------|:--------|:------|:----------|
| **OpenVLA** | 0.0 | 1.0 | 31.0 | 71.0 | – |
| **SmolVLA** | 0.0 | 9.4 | 59.6 | 79.4 | 82.6 |

- **SmolVLA** shows superior *sample efficiency* and *energy efficiency*, making it ideal for real-time deployment.  
- **OpenVLA** demonstrates stronger *transfer learning* and *semantic generalization* due to its larger VLM backbone.  
- Camera rotation caused total performance collapse in OpenVLA → lack of rotation-invariant features.  
- Action chunking reduced accuracy, highlighting the difficulty of multi-step prediction.

---

### 📈 Training Curves
<img width="963" height="800" alt="train" src="https://github.com/user-attachments/assets/a0f06fe2-5c9b-47eb-8435-4caf1eb83335" />

*Figure 2: Training loss and action accuracy curves for OpenVLA (top) and SmolVLA (bottom) under various fine-tuning regimes.*

---

## ⚡ Energy–Performance Trade-Offs

SmolVLA achieves ~80% task success at **half the energy cost** and training time of OpenVLA.  
Training OpenVLA for full fine-tuning required ~14.5 h on an A100 GPU; SmolVLA only ~7.7 h on an RTX 3060 12 GB.
<img width="875" height="498" alt="energy" src="https://github.com/user-attachments/assets/78b83d00-045f-4862-951b-d9b856c7d677" />

*Figure 3: Energy-performance scatter plot for SmolVLA fine-tuning regimes. The full fine-tuning configuration achieves the optimal efficiency/performance balance.*

---

## 🧩 Key Findings

- VLAs can **learn efficient robotic behaviors** with very few demonstrations.  
- Compact architectures like **SmolVLA** provide a viable balance between performance and deployability.  
- Larger models like **OpenVLA** show **better transfer and semantic reasoning** but require more compute.  
- Fine-tuning on spatial benchmarks improves emergent reasoning across unseen manipulation tasks.

---

