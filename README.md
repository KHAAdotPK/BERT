# Training Session Log — February 17, 2026

#### ADHD Workbench Note

> "The best way to understand a black box is to build one."

---

## What I Ran This Morning

```bash
./main.exe infer 10 learning-rate 0.001 epochs 2
```

2 epochs. 50,000 training steps. 25,000 examples per epoch.  
Same codebase as before — **no third layer yet**. This was a baseline run before I make the architectural change.

---

## Training Output

```
Epoch: 1 of 2 epochs.
Step: 1000  | Average Loss: 4.41676
Step: 5000  | Average Loss: 4.16982
Step: 10000 | Average Loss: 4.05954
Step: 15000 | Average Loss: 3.95468
Step: 20000 | Average Loss: 3.88556
Step: 25000 | Average Loss: 3.82170

Epoch: 2 of 2 epochs.
Step: 26000 | Average Loss: 3.81026
Step: 30000 | Average Loss: 3.77055
Step: 35000 | Average Loss: 3.72725
Step: 40000 | Average Loss: 3.69135
Step: 45000 | Average Loss: 3.66189
Step: 50000 | Average Loss: 3.63510
```

---

## Loss Analysis

### Per-Epoch Breakdown

| Epoch | Start Loss | End Loss | Improvement | % Change |
|-------|-----------|----------|-------------|----------|
| **1** | 4.42 | 3.82 | -0.60 | 13.6% |
| **2** | 3.82 | 3.64 | -0.18 | 4.7% |
| **Total** | 4.42 | 3.64 | **-0.78** | **17.6%** |

### What the Loss Curve Tells Me

**Epoch 1** was the heavy lifting phase — the model dropped 0.60 loss points as it captured the primary patterns in the corpus. This is normal. First-epoch learning is always the steepest.

**Epoch 2** showed diminishing returns (0.18 improvement vs 0.60), which is also expected. The model is refining, not discovering. This is healthy convergence behaviour.

The loss decreased at every single 1000-step checkpoint across both epochs. No spikes. No explosions. No divergence. The training was completely clean.

---

## Inference Results

Test input — line 10 of the training corpus:

```
joint-pain  shortness-of-breath  runny-nose
```

Encoder output (pre-trained Skip-gram embeddings, frozen):
```
-2.54968  1.95186  0.159263 -2.89801  0.969118 -0.393037 -0.0821615  1.79531
-1.61708  2.53770  1.12951   0.57520  0.920301  0.377153 -1.11113    2.67982
-1.21579  1.63847 -0.318984  0.611676 0.014152 -1.06068   1.39752    2.34640
0 0 0 0 0 0 0 0   (padding)
0 0 0 0 0 0 0 0   (padding)
0 0 0 0 0 0 0 0   (padding)
0 0 0 0 0 0 0 0   (padding)
```

### Top-5 Predictions Per Token

**Token: `joint-pain`**
```
#1  swelling       (43.0%)  
#2  weight-gain    (40.3%)  
#3  runny-nose     (13.1%)  
#4  tremors        (8.4%)   
#5  abdominal-pain (7.4%)   
```

**Token: `shortness-of-breath`**
```
#1  appetite-loss  (24.6%)  
#2  muscle-pain    (14.8%)  
#3  sneezing       (10.9%)  
#4  cough          (7.2%)   
#5  weight-gain    (7.1%)   
```

**Token: `runny-nose`**
```
#1  sneezing       (37.8%)  
#2  sweating       (31.4%)  
#3  joint-pain     (27.5%)  
#4  chest-pain     (18.7%)  
#5  headache       (13.6%)  
```

---

## Why These Predictions Are Interesting

### `runny-nose → sneezing` (37.8%)

This is the strongest and most medically correct association I have seen from this model so far. Runny nose and sneezing are textbook common-cold co-symptoms. The model learned this entirely from statistical co-occurrence in 25,000 lines of symptom data, with no external medical knowledge injected. It figured it out purely from context.

### `shortness-of-breath → cough` (7.2%)

This is a respiratory family association — both symptoms belong to the same clinical category (respiratory tract). The model is grouping symptoms by physiological system, which is more sophisticated than simple frequency counting.

### `joint-pain → swelling + weight-gain`

These are legitimate joint-related co-symptoms. Swelling at 43% is a strong, confident prediction. The model is not just guessing the most common word in the vocabulary — it is making contextual inferences.

### No Self-Predictions

In my 5-epoch run, the model started predicting the input token itself (`joint-pain → joint-pain`). That was a sign of overfitting. This run shows none of that. Every prediction is a different token from the input, which means the model is still generalising rather than memorising.

---

## Comparison Across All Training Sessions

I have now run this model at various epoch counts. Here is the full picture:

| Session | Epochs | Final Loss | Top-1 (joint-pain) | Confidence | Self-Predict? | Verdict |
|---------|--------|-----------|---------------------|------------|----------------|---------|
| Run 1 | 1 | 3.52 | insomnia | Low | No | ❌ Undertrained |
| Run 2 | 2 | 3.64 | **swelling** | **43%** | **No** | ✅ **Best** |
| Run 3 | 3 | 3.57 | weight-loss | 56% | No | ✅ Good |
| Run 4 | 5 | 3.45 | joint-pain (self) | 8% | **Yes** | ⚠️ Saturating |
| **This run** | 2 | 3.64 | **swelling** | **43%** | **No** | ✅ **Best** |

### The Paradox I Noticed

Lower loss does not always mean better predictions.

The 5-epoch run achieved the lowest loss (3.45) but produced the worst inference quality — the model started predicting the input token itself and showed very low confidence across all predictions. The 2-epoch run has a higher loss (3.64) but produces more meaningful, diverse, and confident predictions.

This is a textbook demonstration of the difference between training loss and generalisation quality. The model with the lowest training loss was not the most useful model.

### What This Means for My Architecture

My current 2-layer FFN with `d_model=8` has a natural capacity limit. Based on these experiments, I can roughly map it out:

```
Optimal zone:       2-3 epochs  (50k-75k steps)
Diminishing returns: 4 epochs   (100k steps)
Saturation/overfitting: 5+ epochs (125k+ steps)
```

The model runs out of things to learn at around 3 epochs. After that, it starts memorising rather than generalising. This is not a bug in the training loop — it is a fundamental limitation of the model capacity.

---

## A Bug I Noticed: Missing Steps

Looking at the epoch 2 output carefully:

```
Step: 36000 | Average Loss: 3.71925
Step: 38000 | Average Loss: 3.70442   ← Step 37000 missing
...
Step: 46000 | Average Loss: 3.65605
Step: 48000 | Average Loss: 3.64513   ← Step 47000 missing
```

Steps 37000 and 47000 never printed. This is a logging bug, not a training bug — the loss trajectory is smooth through these gaps, which means training continued correctly. The `counter % 1000 == 0` check probably has an alignment issue at the epoch boundary.

I need to verify that my global counter is never reset between epochs. It should increment continuously from 1 to 50000, never restart at zero when epoch 2 begins. If it resets, certain multiples of 1000 would never be reached.

TODO: Fix this before the 3-layer implementation so I have clean logs to compare against.

---

## Why I Am Adding a Third Layer

This training session is the empirical justification for adding a third hidden layer.

The 2-layer model has proven it can learn. It makes medically coherent predictions. The backpropagation is correct. The gradient accumulation works. The training infrastructure is solid.

But the model saturates too quickly. By epoch 3, it stops discovering new patterns. By epoch 5, it starts memorising. A third hidden layer will:

- Increase representational capacity (more complex patterns extractable)
- Delay the saturation point (more epochs before overfitting)
- Allow the model to learn higher-order relationships between symptoms
- Potentially push the loss below 3.40 while maintaining prediction diversity

The target for the 3-layer run: achieve loss below 3.64 at 2 epochs with no self-predictions and maintained or improved prediction diversity.

---

## What the 3-Layer Declaration Needs

I already sketched the class declaration. The key additions beyond the current 2-layer architecture:

**New weights and biases:**
```cpp
Collective<E> w_hidden_2;  // [d_model x d_model]
Collective<E> b_hidden_2;  // [1 x d_model]
```

**New gradient accumulators** (this was missing from my first draft):
```cpp
Collective<E> dHidden1_dw;  // renamed from dHidden_dw
Collective<E> dHidden1_db;  // renamed from dHidden_db
Collective<E> dHidden2_dw;  // new
Collective<E> dHidden2_db;  // new
```

**New forward pass cache variables** (also missing from first draft):
```cpp
Collective<E> last_hidden_1_raw;        // Z1
Collective<E> last_hidden_1_activated;  // H1 = ReLU(Z1)
Collective<E> last_hidden_2_raw;        // Z2
Collective<E> last_hidden_2_activated;  // H2 = ReLU(Z2)
```

The backward pass then follows the chain rule through one extra layer. The pattern is identical to the existing layer — just applied twice with the correct cached tensors.

---

## Current State of the Codebase

| Component | Status |
|-----------|--------|
| Forward propagation (2-layer) | ✅ Correct and validated |
| Backward propagation (2-layer) | ✅ Correct and validated |
| Cross-entropy loss | ✅ Numerically stable |
| Gradient accumulation (16 steps) | ✅ Correct |
| Multi-epoch training | ✅ Working |
| Gradient clipping | ❌ Not yet implemented |
| Checkpoint saving | ❌ Not yet implemented |
| Validation set split | ❌ Not yet implemented |
| Third hidden layer | 🔄 In progress |

### Gradient Clipping — Still Not Implemented

I know. I keep not implementing it. The training has been stable without it at LR=0.001, but I had one session earlier that exploded badly when I was running at LR=0.01. The clipping code is straightforward — it is on the list before I push to any larger dataset or higher learning rate.

```cpp
// The 20 lines I keep not writing:
E total_norm = sqrt(sum of all squared gradient values);
if (total_norm > 1.0) {
    E scale = 1.0 / total_norm;
    // multiply all gradient accumulators by scale
}
```

---

## Next Steps

1. **Fix the missing step logging bug** — verify global counter is never reset at epoch boundaries
2. **Implement the third hidden layer** — full forward pass, backward pass, weight update, accumulator reset
3. **Run baseline comparison** — same 2-epoch run with 3-layer model, compare loss and inference quality
4. **Finally implement gradient clipping** — it needs to happen before I try higher learning rates
5. **Add checkpoint saving** — losing 125k steps to a crash would be annoying

---

## Hardware & Environment

```
OS:       Windows 10
CPU:      Intel Pentium
RAM:      16 GB
Training: CPU-only (no GPU)
Language: C++ (zero external ML dependencies)
Build:    F:\BERT\usage> cl /EHsc ./src/main.cpp
```

---

*Session logged: February 17, 2026*  
*Next session: 3-layer MLM head implementation*  
*Contact: Q@hackers.pk*
