#### **ADHD Workbench Note**

> "The best way to understand a black box is to build one." This repo is part of a larger ecosystem of custom-built neural components designed to prove that the most complex AI models are just a series of simple, elegant mathematical loops.

# BERT from Scratch (C++ Implementation)

This repository contains a high-performance implementation of the **BERT (Bidirectional Encoder Representations from Transformers)** architecture built entirely from scratch in **C++**.

The goal of this project is to create a lightweight, "dependency-free" AI engine capable of training and inference on consumer-grade hardware (CPU-focused) with maximum memory efficiency.

## 🚀 Current Project Status (V1.0 Skeleton)

We have successfully implemented the core **Masked Language Modeling (MLM)** head and training infrastructure. The model is currently demonstrating learning capabilities on a sample corpus of 25,000 lines.

### What has been implemented so far:

* **MLM Head Architecture:**
* **Linear Projection Layer:** Maps encoder hidden states () to the vocabulary size.
* **Bias Integration:** Trainable bias vectors for every token in the vocabulary.
* **The Training Loop:**
* **Forward Pass:** Complete path from Encoder Output () to Logits and Probabilities.
* **Cross-Entropy Loss:** Mathematical implementation for comparing predictions against ground-truth labels (ignoring non-masked tokens).
* **Backpropagation:** Full manual implementation of gradients for weights () and biases ().
* **Stochastic Gradient Descent (SGD):** Real-time weight updates using a configurable learning rate.
* **Data Pipeline:**
* Support for the BERT noise strategy: **[MASKED]** (80%), **[RANDOM]** (10%), and **[KEEP]** (10%).
* Tokenization and vocabulary mapping for a custom corpus.

## 📊 Observations

The model currently shows healthy training behavior. Initial loss values start around **3.3 - 4.5** (mathematically consistent with a random start for a 28-token vocab) and have been observed dropping as low as **0.03** on recognized patterns, proving that the gradient descent logic is working.

## 🏗 Next Steps: Optimization & Inference

We are currently moving from a "functional skeleton" to a "stable product." The immediate focus is on the following two milestones:

### 1. Moving Average Loss Tracking (Metric Health)

To better understand the global learning curve, we are implementing a **Moving Average Loss** (calculated over every 1,000 lines).

* **Goal:** Filter out the "noise" of individual lines (stochastic jumps) to visualize the true downward trend of the model's error.
* **Implementation:** A sliding window accumulator to provide a stable "Health Score" for the training process.

### 2. Argmax & Top-K Decoding (Inference Implementation)

We are building the logic to transform raw numerical outputs (Logits) into human-readable predictions.

* **Argmax:** Logic to find the single most likely token index for a `[MASK]`.
* **Top-K Selection:** Implementing a sorting mechanism to extract the **Top 5** most probable words. This is the final step required to power a visual demo where the model fills in the blanks.

## 🛠 Tech Stack

* **Language:** Pure C++ (Templates)
* **Math Library:** Internal `Numcy` Engine (No reliance on heavy frameworks like PyTorch or TensorFlow)
* **Hardware Target:** Optimized for high-speed CPU inference.

---

## **License**

This project is governed by a license, available in the accompanying `LICENSE` file.  
Please refer to it for complete licensing details.