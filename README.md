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
---


### 📈 Training Curves
<img width="963" height="800" alt="train" src="https://github.com/user-attachments/assets/a0f06fe2-5c9b-47eb-8435-4caf1eb83335" />

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

### 🔍 Zero-Shot Generalization

To evaluate generalization to unseen environments, both pretrained models were tested directly on **LIBERO-Spatial** and **LIBERO-Object** without any fine-tuning.

- In **pure zero-shot**, both models failed to complete tasks successfully due to the real-to-sim gap and differing camera configurations.  
  Their motion patterns appeared random, indicating no spatial understanding of the new environment.

- After fine-tuning on **LIBERO-Spatial**, **OpenVLA** demonstrated *emergent semantic understanding*: it began moving toward target objects and sometimes grasped them correctly on unseen **LIBERO-Object** tasks where it achieved partial success without retraining.

<img width="968" height="488" alt="object" src="https://github.com/user-attachments/assets/29695f1f-707b-48cb-884c-290de0fceafd" />

- **SmolVLA**, while faster and more efficient, lacked this emergent transfer behavior.  
  Its smaller 2 B backbone limited its ability to generalize to unseen object manipulation tasks.

- With only 10 episodes of fine-tuning in the **LIBERO-Object** benchmark (1 episode per task) we see that OpenVLA learns the new task much more effectively, utilizing its *emergent semantic understanding* capabilities.

| Model | Fine-tuning | Success Rate (%) on LIBERO-Object |
|:------|:-------------|:---------------------------------:|
| **OpenVLA** | Zero-shot | 0.0 |
| | + 10 Episodes | 5.5 |
| **SmolVLA** | Zero-shot | 0.0 |
| | + 10 Episodes | 0.0 |

These findings highlight that **large-scale VLAs** like OpenVLA retain richer visual-semantic priors, while **smaller VLAs** like SmolVLA require additional examples to transfer knowledge across tasks and embodiments.

---

## ⚡ Energy–Performance Trade-Offs

SmolVLA achieves ~80% task success at **half the energy cost** and training time of OpenVLA.  
Training OpenVLA for full fine-tuning required ~14.5 h on an A100 GPU; SmolVLA only ~7.7 h on an RTX 3060 12 GB.

<img width="875" height="498" alt="energy" src="https://github.com/user-attachments/assets/78b83d00-045f-4862-951b-d9b856c7d677" />

---

## 🧩 Key Findings

- VLAs can **learn efficient robotic behaviors** with very few demonstrations.  
- Compact architectures like **SmolVLA** provide a viable balance between performance and deployability.  
- Larger models like **OpenVLA** show **better transfer and semantic reasoning** but require more compute.  
- Fine-tuning on spatial benchmarks improves emergent reasoning across unseen manipulation tasks.

---

