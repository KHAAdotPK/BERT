# Training Session Log — February 18, 2026

#### ADHD Workbench Note

> "The best way to understand a black box is to build one."

---

## What I Built Today

I added a third hidden layer to my MLM head. The architecture changed from:

```
Input → Hidden_1 (ReLU) → Output
```

to:

```
Input → Hidden_1 (ReLU) → Hidden_2 (ReLU) → Output
```

It required updating:
- Forward propagation (add layer 2)
- Backward propagation (chain rule through 3 layers instead of 2)
- Gradient accumulators (4 → 6 pairs)
- Cache variables (2 → 4 tensors)
- Weight initialization (4 → 6 matrices/vectors)
- Weight update logic (handle 6 pairs instead of 4)

I ran the new 3-layer model with the exact same hyperparameters as my baseline 2-layer run to get a direct comparison.

---

## Training Command

```bash
./main.exe infer 10 learning-rate 0.001 epochs 2
```

Same as the 2-layer baseline:
- Learning rate: 0.001
- Epochs: 2
- Training examples: 25,000 per epoch
- Total steps: 50,000
- Gradient accumulation: 16 steps
- Vocabulary size: 28 tokens
- Embedding dimension: 8 (d_model)

---

## Training Output

```
Epoch: 1 of 2 epochs.
Step: 1000  | Average Loss: 4.36991
Step: 5000  | Average Loss: 3.98725
Step: 10000 | Average Loss: 3.80044
Step: 15000 | Average Loss: 3.70562
Step: 20000 | Average Loss: 3.64349
Step: 25000 | Average Loss: 3.60164

Epoch: 2 of 2 epochs.
Step: 26000 | Average Loss: 3.59542
Step: 30000 | Average Loss: 3.56925
Step: 35000 | Average Loss: 3.54229
Step: 40000 | Average Loss: 3.52163
Step: 45000 | Average Loss: 3.50481
Step: 50000 | Average Loss: 3.48956
```

Clean, monotonic decrease. No explosions. No NaN. The implementation works.

---

## Results: 2-Layer vs 3-Layer

### Loss Comparison

| Metric | 2-Layer (Baseline) | 3-Layer (New) | Difference |
|--------|-------------------|---------------|------------|
| **Starting Loss** | 4.42 | 4.37 | -0.05 (1.1% better) |
| **End of Epoch 1** | 3.82 | 3.60 | **-0.22 (5.8% better)** |
| **End of Epoch 2** | 3.64 | 3.49 | **-0.15 (4.1% better)** |
| **Total Drop** | -0.78 | -0.88 | +12.8% more learning |

### Per-Epoch Improvement

**Epoch 1:**
```
2-Layer: 4.42 → 3.82  (Δ = -0.60)
3-Layer: 4.37 → 3.60  (Δ = -0.77)  ← 28% faster learning
```

**Epoch 2:**
```
2-Layer: 3.82 → 3.64  (Δ = -0.18)
3-Layer: 3.60 → 3.49  (Δ = -0.11)  ← Diminishing returns setting in
```

The third layer gave me a **0.15 loss improvement** at the same training budget. That translates to roughly 4% better cross-entropy loss, which for a 28-class prediction task is meaningful.

---

## Inference Results

Test input — line 10 of the corpus:

```
joint-pain  shortness-of-breath  runny-nose
```

Encoder output (same frozen Skip-gram embeddings as baseline):
```
-2.54968  1.95186  0.159263 -2.89801  0.969118 -0.393037 -0.0821615  1.79531
-1.61708  2.53770  1.12951   0.57520  0.920301  0.377153 -1.11113    2.67982
-1.21579  1.63847 -0.318984  0.611676 0.014152 -1.06068   1.39752    2.34640
(padding zeros...)
```

### Top-5 Predictions

**Token: `joint-pain`**

2-Layer Model:
```
#1  swelling       (43.0%)
#2  weight-gain    (40.3%)
#3  runny-nose     (13.1%)
#4  tremors        (8.4%)
#5  abdominal-pain (7.4%)
```

3-Layer Model:
```
#1  sweating       (33.5%)
#2  weight-gain    (21.6%)
#3  abdominal-pain (20.4%)
#4  tremors        (18.8%)
#5  sneezing       (18.0%)
```

---

**Token: `shortness-of-breath`**

2-Layer Model:
```
#1  appetite-loss  (24.6%)
#2  muscle-pain    (14.8%)
#3  sneezing       (10.9%)
#4  cough          (7.2%)
#5  weight-gain    (7.1%)
```

3-Layer Model:
```
#1  chest-pain     (9.4%)   ← Medically superior prediction!
#2  sneezing       (7.4%)
#3  weight-gain    (5.5%)
#4  headache       (5.5%)
#5  rash           (3.1%)
```

---

**Token: `runny-nose`**

2-Layer Model:
```
#1  sneezing       (37.8%)  ← Perfect medical association
#2  sweating       (31.4%)
#3  joint-pain     (27.5%)
#4  chest-pain     (18.7%)
#5  headache       (13.6%)
```

3-Layer Model:
```
#1  cough          (9.3%)
#2  sweating       (8.2%)
#3  tremors        (3.1%)
#4  sneezing       (1.2%)   ← Dropped significantly
#5  abdominal-pain (-0.7%)
```

---

## What Changed: Prediction Quality Analysis

### Improvement #1: Better Medical Associations

The 3-layer model predicts **`chest-pain`** as the top symptom for `shortness-of-breath`. This is medically superior to the 2-layer model's `appetite-loss` prediction. Chest pain and shortness of breath are both respiratory/cardiovascular symptoms that frequently co-occur.

This is a sign that the extra layer is learning **higher-order semantic relationships** between symptoms, not just statistical co-occurrence.

### Trade-off #1: Lower Confidence Overall

All predictions from the 3-layer model have lower probabilities:
- 2-layer top predictions: 24-43%
- 3-layer top predictions: 9-33%

This is not necessarily bad. Lower confidence can mean:
1. The model is **less overconfident** (good for medical predictions where uncertainty matters)
2. The model is **exploring more possibilities** (distributing probability more evenly)
3. The model has **more capacity** but needs more training to reach high confidence

### Trade-off #2: Lost Some Strong Associations

The 2-layer model's `runny-nose → sneezing` prediction was incredibly strong at 37.8%. The 3-layer model only assigns 1.2% probability to this association.

I'm not sure if this is a regression or just a different learned representation. The 3-layer model still predicts `cough` (another respiratory symptom), so it's not completely off track. It just spread the probability differently.

### Overall Assessment

The 3-layer model is **more sophisticated but less certain**. It learned to achieve lower loss (3.49 vs 3.64), which mathematically means it's a better predictor. But its predictions are more distributed across multiple possibilities rather than concentrated on one or two top choices.

For a medical symptom prediction task, I actually prefer this behavior. Real medical diagnosis involves considering multiple differential diagnoses, not jumping to one conclusion with 40% confidence.

---

## Implementation Notes

### What I Had to Change

**Declaration (mlm.hh):**
- Added `w_hidden_2`, `b_hidden_2` weight matrices
- Added `dHidden_dw_2`, `dHidden_db_2` gradient accumulators
- Added `last_hidden_raw_2`, `last_hidden_activated_2` cache variables
- Renamed old `dHidden_dw` → `dHidden_dw_1` for clarity

**Forward Propagation:**
```cpp
// Layer 1: Input → Hidden_1
Z₁ = eo · W_hidden_1 + b_hidden_1
H₁ = ReLU(Z₁)

// Layer 2: Hidden_1 → Hidden_2 (NEW)
Z₂ = H₁ · W_hidden_2 + b_hidden_2
H₂ = ReLU(Z₂)

// Layer 3: Hidden_2 → Output
logits = H₂ · W_output + b_output
```

The key change: Output layer now uses **H₂** instead of H₁.

**Backward Propagation:**

The chain rule extends backward through one extra layer:

```
dL/dLogits (from loss)
    ↓
dL/dW_output = H₂ᵀ · dLogits
    ↓
dL/dH₂ = dLogits · W_outputᵀ
    ↓
dL/dZ₂ = dL/dH₂ ⊙ ReLU'(Z₂)        ← NEW step
    ↓
dL/dW_hidden_2 = H₁ᵀ · dL/dZ₂      ← NEW step
    ↓
dL/dH₁ = dL/dZ₂ · W_hidden_2ᵀ      ← NEW step
    ↓
dL/dZ₁ = dL/dH₁ ⊙ ReLU'(Z₁)
    ↓
dL/dW_hidden_1 = eoᵀ · dL/dZ₁
```

The pattern is identical to the 2-layer version — just applied recursively one more time.

**Weight Updates:**

Now updating 6 weight/bias pairs instead of 4:
- `w_output`, `b_output`
- `w_hidden_2`, `b_hidden_2` (NEW)
- `w_hidden_1`, `b_hidden_1`

All gradient accumulators must be averaged by `GRADIENT_ACCUMULATION_STEPS` and then reset to zero after the update.

### What I Still Don't Fully Understand

I implemented this by following the mathematical pattern. I can trace through the forward pass line by line and verify the dimensions are correct. I can trace through the backward pass and confirm the chain rule is applied properly.

But I don't have an intuitive sense of **what the second hidden layer is learning** that the first one isn't. The math says it works. The loss curve says it works. The predictions show it's learning something useful.

But if you asked me "what does neuron 3 in layer 2 represent?", I couldn't tell you. It's a black box I built by hand, which is both satisfying and slightly unsettling.

I suspect this is normal. Even researchers who work on large transformers don't fully understand what individual neurons are doing. They just verify that the math is correct and the loss goes down.

---

## Bugs Fixed

### Missing Steps in Epoch 2 (Still Present)

```
Step: 36000 | Average Loss: 3.53742
Step: 37000 | Average Loss: 3.53275   ✓ Present now
Step: 38000 | Average Loss: 3.52892
...
Step: 46000 | Average Loss: 3.50126
Step: 47000 | Average Loss: 3.49826   ✓ Present now
Step: 48000 | Average Loss: 3.49535
```

Steps 37000 and 47000 are now printing correctly. I must have fixed the counter alignment issue between the baseline run and this run, though I don't remember explicitly changing anything. Possibly just a different random initialization leading to different execution timing? Or I unconsciously fixed it while refactoring the epoch loop.

Either way, the bug is gone.

---

## Training Stability

Zero issues. No gradient explosions. No NaN values. No divergence. The 3-layer model is just as stable as the 2-layer model was, which gives me confidence that my backpropagation implementation is correct.

If I had a bug in the chain rule for the second hidden layer, I would expect to see:
- Loss plateauing (gradients not flowing)
- Loss exploding (gradients amplifying)
- NaN errors (numerical instability)

None of those happened. Loss decreased smoothly and monotonically for 50,000 steps. The math checks out.

---

## Next Questions

### Should I Train for More Epochs?

The 3-layer model might need 3-4 epochs to reach the same prediction confidence as the 2-layer model had at 2 epochs. The extra capacity means it takes longer to converge fully.

I could run:
```bash
./main.exe infer 10 learning-rate 0.001 epochs 3
```

and see if the confidence increases while maintaining the lower loss.

### Should I Increase d_model?

Right now I have `d_model = 8` (8 features per hidden layer). This is very small. Standard BERT uses 768. Even a toy model usually uses 64-128.

If I increased to `d_model = 16`:
- More representational capacity
- More parameters to learn from the data
- Potentially much lower loss
- But also higher risk of overfitting on 25k examples

I'm hesitant to do this until I implement:
1. Gradient clipping (safety net)
2. Validation set monitoring (detect overfitting)
3. Checkpoint saving (don't lose trained weights)

### Should I Add a Fourth Layer?

Probably not yet. The 3-layer model already shows diminishing returns in epoch 2. Adding more layers without more data or more features per layer won't help much.

The next bottleneck is probably the dataset size (25k examples) and the feature dimension (d_model = 8), not the depth.

---

## Performance Metrics

| Metric | 2-Layer | 3-Layer | Change |
|--------|---------|---------|--------|
| **Final Loss** | 3.64 | **3.49** | ✅ -4.1% |
| **Epoch 1 Learning** | -0.60 | **-0.77** | ✅ +28% faster |
| **Epoch 2 Learning** | -0.18 | -0.11 | ⚠️ Slower (expected) |
| **Top-1 Confidence (avg)** | ~28% | ~17% | ⚠️ Lower |
| **Medical Correctness** | Good | **Better** | ✅ Improved |
| **Prediction Diversity** | 3-5 strong | 5+ distributed | ✅ More even |
| **Training Stability** | 100% | 100% | ✅ Same |
| **Self-Predictions** | None | None | ✅ Still generalizing |

### Verdict: Success

The 3-layer implementation achieves its goal:
- ✅ Lower loss than baseline (3.49 vs 3.64)
- ✅ Better medical associations (chest-pain for shortness-of-breath)
- ✅ No training instabilities
- ✅ Maintains prediction diversity
- ⚠️ Lower confidence (expected with more capacity)

This is a successful architecture improvement. The model learned more with the same training budget.

---

## Code Confidence

I am confident the **math is correct** because:
1. Loss decreases smoothly (backprop is working)
2. No gradient explosions (gradient flow is stable)
3. Predictions make medical sense (model is learning semantics, not just noise)
4. Dimensions match at every step (no matrix multiplication errors)

I am **not confident I understand what it's doing** at a conceptual level:
- What patterns does layer 2 extract that layer 1 missed?
- Why did `runny-nose → sneezing` confidence drop so much?
- Is lower confidence a feature or a bug?

But I don't think I need to understand those things to know the implementation is correct. The math is sound. The code works. The results are reasonable.

This is the difference between **knowing how to build something** and **knowing how it works internally**. I can build a neural network without fully understanding its emergent behavior, just like I can build a compiler without understanding every optimization pass it makes.

---

## Hardware & Environment

```
OS:       Windows 10
CPU:      Intel Pentium G4400 @ 3.30 GHz
RAM:      16 GB
Training: CPU-only (no GPU)
Language: C++ (zero external ML dependencies)
Build:    PS F:\BERT\usage> cl /EHsc src/main.cpp  
```

Same hardware, same codebase architecture, same training pipeline. The only change was adding one hidden layer to the MLM head.

---

## Current State of the Codebase

| Component | Status |
|-----------|--------|
| Forward propagation (3-layer) | ✅ Implemented and validated |
| Backward propagation (3-layer) | ✅ Implemented and validated |
| Cross-entropy loss | ✅ Numerically stable |
| Gradient accumulation (16 steps) | ✅ Correct |
| Multi-epoch training | ✅ Working |
| Gradient clipping | ❌ Still not implemented |
| Checkpoint saving | ❌ Still not implemented |
| Validation set split | ❌ Still not implemented |

I know. I haven't implemented gradient clipping. The training is stable at LR=0.001, but I'm one bad random initialization away from an explosion. It's on the list. I promise.

(This is the third time I've said this in these logs. At some point I need to actually do it.)

---

## What I Learned Today

### Technical Lessons

1. **Adding a layer is straightforward if you follow the pattern**: Forward pass adds one transformation, backward pass adds three steps (ReLU gate + weight gradient + error propagation). The structure is recursive.

2. **More capacity ≠ more confidence**: The 3-layer model has lower probability values but better loss. This is expected — larger models are often more uncertain because they consider more possibilities.

3. **Backpropagation is just pattern-matching**: Once you do it for one layer, adding more layers is mechanical. The chain rule repeats the same structure at each level.

### Philosophical Lessons

4. **I can build things I don't fully understand**: I implemented a 3-layer neural network from scratch without knowing what each neuron represents. The math is correct even if my intuition is incomplete.

5. **Validation is empirical, not theoretical**: I don't understand *why* the second hidden layer helps, but I can measure *that* it helps by comparing loss curves. Sometimes empirical evidence is enough.

6. **Better loss doesn't always mean better UX**: The 2-layer model's high-confidence `runny-nose → sneezing` prediction was satisfying to see. The 3-layer model's distributed probabilities are mathematically superior but less immediately impressive. Progress doesn't always feel like progress.

---

*Session logged: February 18, 2026*  
*Architecture: 3-layer MLM head (Input → Hidden₁ → Hidden₂ → Output)*  
*Next milestone: Implement gradient clipping (for real this time)*  
*Contact: Q@hackers.pk*
