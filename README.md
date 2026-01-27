## BERT (Bidirectional Encoder Representations from Transformers): from-Scratch Medical Symptom Analysis

This repository contains a **Masked Language Model (MLM)** implementation built from the ground up in **C++**. The project focuses on processing medical symptom sequences using bidirectional context; allowing the model to "understand" the relationship between clinical terms by predicting hidden tokens.

### 🚀 The Mission

While most implementations rely on high level libraries (like HuggingFace), this project is a **"Mechanical Deep Dive."** Every moving part; from the binary data pipeline to the Cross-Entropy loss is implemented manually to master the internal architecture of the Transformer Encoder.

#### Hybrid Training: Skip-gram Warm-up → BERT Pre-training Initialization

A key efficiency and performance feature of this implementation is **hybrid training**.  

Instead of starting the Transformer encoder embeddings from pure random initialization (which can lead to slow convergence on small/custom datasets), I first train a **shallow Skip-gram model** on the same medical symptom corpus to learn high quality static word vectors. These pre-trained Skip-gram embeddings are then used to **initialize** the BERT embedding layer.

This is a classic **pre-training initialization** strategy:  
- The shallow model (Skip-gram) quickly captures basic co-occurrence semantics.  
- These informed weights "warm up" the deep bidirectional model (BERT-style MLM), helping it converge faster and reach better contextual representations with fewer epochs / less compute.

This technique is especially valuable on domain-specific data (like medical symptoms) where general purpose pre-trained embeddings may not exist or may underperform.

### 🧠 Architectural Shift: Bidirectional vs. Autoregressive

Unlike my previous [Transformer-Encoder-Decoder](https://github.com/KHAAdotPK/Transformer-Encoder-Decoder) project, this BERT implementation is **NOT Autoregressive**.

* **The Old Way (Autoregressive):** The model was forced to look only at the past (using a [Triangular Causal Mask](https://github.com/KHAAdotPK/MachineLearning/blob/main/outline_of_constructing_decoder_input_pipeline.md)) to predict the next token. It was a "one-way" street.
* **The BERT Way (Bidirectional):** This model uses an "Open-Room" approach. Every token can see every other token in the sequence (past and future) simultaneously.

By removing the causal mask, the Encoder can create a much richer representation of a symptom. For example, in the sequence `[High, [MASK], and, Chills]`, the model uses the word "Chills" (the future) to help identify that the `[MASK]` is likely "Fever."

### 🛠 What I Have Done

* **Skip-gram Integration:** Successfully trained word vectors to provide the model with a "semantic starting point."
* **Binary Data Pipeline:** Developed a high-performance C++ `ifstream` loader to inject pre-trained weights into the system memory.
* **Contextual Architecture Design:** Designed the project to interface with a separate [Transformer-Encoder-Decoder](https://github.com/KHAAdotPK/Transformer-Encoder-Decoder) library, maintaining a clean, modular repository structure.
* **MLM Head Logic:** Defined the mathematical projection from the `d_model` (8 dimensions) to the `vocab_size` (28 dimensions) to generate word probabilities.

### 🏗 What I Am Doing (Work-in-Progress)

* **The "Damager" (Masking Engine):** Implementing the stochastic 15% masking rule (80% [MASK], 10% Random, 10% Unchanged) to generate training pairs from 25,000 lines of symptoms.
* **Pipeline Wiring:** Connecting the raw symptom text files to the embedding lookup table to create the initial  input tensor.
* **Softmax & Loss Implementation:** Writing the C++ loops for the Softmax probability distribution and the Cross-Entropy "Pain" signal for backpropagation.

### ⏭ What is Next

1. **Lego Brick Integration:** Pull the Transformer Encoder layers from the sister repository into this pipeline.
2. **Training Loop:** Run the first epoch of MLM training to see the "Pain" (Loss) value decrease as the model learns to fill in the blanks.
3. **Hyper Parameter Tuning:** Experiment with averaging  and  weights to see the effect on contextual understanding.
4. **Visualization:** Create a console-based "Live View" where you can see the model's top-3 guesses for a masked medical term in real-time.

### 📊 Technical Specs

* **Tokens per line:** 7
* **Embedding Dim:** 8
* **Vocabulary Size:** 28
* **Language:** C++ (Standard 17/20)
* **Dataset:** 25,000 medical symptom sequences

#### **ADHD Workbench Note**

> "The best way to understand a black box is to build one." This repo is part of a larger ecosystem of custom-built neural components designed to prove that the most complex AI models are just a series of simple, elegant mathematical loops.

---

## **License**

This project is governed by a license, available in the accompanying `LICENSE` file.  
Please refer to it for complete licensing details.