#### BERT is encoder-only. There is no decoder in the BERT architecture. The full picture is:

```text
Input tokens
     ↓
Embedding Layer
     ↓
Transformer Encoder (stack of N layers, each with Multi-Head Attention + FFN)
     ↓
Encoder Output (contextual embeddings)
     ↓
MLM Head (your 3-layer FFN)
     ↓
Logits → Softmax → Cross-Entropy Loss
```

### Model is too small (396 parameters)  

#### 📊 The calculation:

```cpp
// The 3-layer MLM head has 6 weight matrices/vectors:

// Layer 1: Hidden layer 1
W_hidden_1: [d_model x d_model] = [8 x 8] = 64 parameters
b_hidden_1: [1 x d_model]       = [1 x 8] = 8 parameters

// Layer 2: Hidden layer 2  
W_hidden_2: [d_model x d_model] = [8 x 8] = 64 parameters
b_hidden_2: [1 x d_model]       = [1 x 8] = 8 parameters

// Layer 3: Output layer
W_output:   [d_model x vocab_size] = [8 x 28] = 224 parameters
b_output:   [1 x vocab_size]       = [1 x 28] = 28 parameters

// Total:
64 + 8 + 64 + 8 + 224 + 28 = 396 parameters
```

---

#### Breaking it down:

##### Layer 1 (Input → Hidden_1):
```text
W_hidden_1 = 8 rows × 8 columns = 64 weights
b_hidden_1 = 8 bias values       = 8 biases
Subtotal: 72 parameters
```

#### Layer 2 (Hidden_1 → Hidden_2):
```text
W_hidden_2 = 8 rows × 8 columns = 64 weights  
b_hidden_2 = 8 bias values       = 8 biases
Subtotal: 72 parameters
```text

#### Layer 3 (Hidden_2 → Output):
```text
W_output = 8 rows × 28 columns = 224 weights
b_output = 28 bias values       = 28 biases
Subtotal: 252 parameters
```

#### Grand Total:
```text
72 + 72 + 252 = 396 trainable parameters
```

---

#### Why this matters:

**For comparison:**

| Model | Parameters |
|-------|-----------|
| **Your 3-layer MLM** | **396** |
| BERT-tiny (research) | ~4.4 million |
| BERT-base (standard) | ~110 million |
| GPT-2 (small) | ~117 million |
| GPT-3 | ~175 **billion** |

This model is still **0.000036%** of BERT-base's parameters.

---

#### Why GPU doesn't help:

With only 396 parameters:
- Matrix multiplications are tiny (8×8, 8×28)
- GPU overhead (kernel launch, data transfer) dominates
- CPU can multiply 8×8 matrices in microseconds
- GPU would spend more time setting up than computing

**GPU becomes useful around 100,000+ parameters** (about 250x larger than your model).

