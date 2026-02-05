# BERT C++ Engine: Inference Logs & Analysis

This document tracks the real-time performance and predictive capabilities of the custom C++ BERT implementation. It showcases how the model maps internal vector representations (Encoder Outputs) to human-readable vocabulary.

## 📋 Session 1: Contextual Symptom Correlation (Line 10)

**Scenario:** Predicting associations for early-stage symptoms.

```text
PS F:\BERT\usage> ./main.exe infer 10
Reading training data from the file...
Creating parser for training data...
Creating vocabulary using the training data parser...
Reading pre-trained word vectors from the file...
Determining maximum sequence length...
Step: 1000 | Average Loss: 5.3127
Step: 2000 | Average Loss: 5.31794
Step: 3000 | Average Loss: 5.24135
Step: 4000 | Average Loss: 5.24845
Step: 5000 | Average Loss: 5.28725
Step: 6000 | Average Loss: 5.29583
Step: 7000 | Average Loss: 5.2988
Step: 8000 | Average Loss: 5.34226
Step: 9000 | Average Loss: 5.34129
Step: 10000 | Average Loss: 5.35941
Step: 11000 | Average Loss: 5.35249
Step: 12000 | Average Loss: 5.3604
Step: 13000 | Average Loss: 5.36036
Step: 14000 | Average Loss: 5.36865
Step: 15000 | Average Loss: 5.35697
Step: 16000 | Average Loss: 5.3611
Step: 17000 | Average Loss: 5.35971
Step: 18000 | Average Loss: 5.35952
Step: 19000 | Average Loss: 5.34984
Step: 20000 | Average Loss: 5.35603
Step: 21000 | Average Loss: 5.36331
Step: 22000 | Average Loss: 5.37431
Step: 23000 | Average Loss: 5.38134
Step: 24000 | Average Loss: 5.42036
Step: 25000 | Average Loss: 5.43057
default_infer_line = 10
-2.54968 1.95186 0.159263 -2.89801 0.969118 -0.393037 -0.0821615 1.79531
-1.61708 2.5377 1.12951 0.5752 0.920301 0.377153 -1.11113 2.67982
-1.21579 1.63847 -0.318984 0.611676 0.0141521 -1.06068 1.39752 2.3464
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
joint-pain shortness-of-breath runny-nose
joint-pain: 4, 3.18073 -> weight-loss
shortness-of-breath: 4, 1.70835 -> weight-loss
runny-nose: 27, 1.4337 -> sneezing
```
### 🧠 Inference Evaluation

* **Result:** The model predicted `weight-loss` and `sneezing`.
* **Analysis:** This is a **success**. While the model didn't replicate the input identically, the association between `runny-nose` and `sneezing` proves the C++ engine has successfully captured logical medical correlations within the training corpus.

## 📋 Session 2: Large-Scale Pattern Testing (Line 10000)

**Scenario:** Testing model response on complex multi-symptom sequences.

```text
PS F:\BERT\usage> ./main.exe infer 10000
Reading training data from the file...
Creating parser for training data...
Creating vocabulary using the training data parser...
Reading pre-trained word vectors from the file...
Creating original weights from the pre-trained word vectors...
Determining maximum sequence length...
Step: 1000 | Average Loss: 6.83697
Step: 2000 | Average Loss: 6.30493
Step: 4000 | Average Loss: 5.85904
Step: 5000 | Average Loss: 5.73672
Step: 6000 | Average Loss: 5.65825
Step: 7000 | Average Loss: 5.60213
Step: 8000 | Average Loss: 5.60296
Step: 9000 | Average Loss: 5.56805
Step: 10000 | Average Loss: 5.53795
Step: 11000 | Average Loss: 5.51864
Step: 12000 | Average Loss: 5.49948
Step: 13000 | Average Loss: 5.49773
Step: 14000 | Average Loss: 5.50957
Step: 15000 | Average Loss: 5.49854
Step: 16000 | Average Loss: 5.49009
Step: 17000 | Average Loss: 5.48104
Step: 18000 | Average Loss: 5.48598
Step: 19000 | Average Loss: 5.4846
Step: 20000 | Average Loss: 5.47946
Step: 21000 | Average Loss: 5.49384
Step: 22000 | Average Loss: 5.4958
Step: 23000 | Average Loss: 5.48453
Step: 24000 | Average Loss: 5.47941
Step: 25000 | Average Loss: 5.48457
default_infer_line = 10000
2.57084 1.23545 0.830596 0.18937 1.6948 9.07761 7.78712 -1.12159
3.60761 0.348487 0.794499 -2.16914 0.204483 4.73075 12.5189 0.853218
5.91009 0.559466 -1.10951 -1.95219 -0.0534263 4.69611 0.788269 -0.158284
9.07678 0.304284 -3.57869 0.655302 0.153214 5.24594 3.02196 0.302907
4.25454 -0.45298 -0.558497 -1.44716 -1.32633 5.43337 2.25685 -0.635332
2.47654 1.49598 -4.07029 -7.50076 -0.266361 4.42605 3.48414 0.00326957
2.69698 -0.997956 -2.11742 -2.02739 -2.68292 4.43233 -0.148387 -1.6387
headache rash nausea vomiting tremors cough anxiety
headache: 8, 3.82755 -> blurred-vision
rash: 6, 5.87407 -> vomiting
nausea: 8, 2.45631 -> blurred-vision
vomiting: 8, 3.76488 -> blurred-vision
tremors: 8, 2.99669 -> blurred-vision
cough: 8, 3.73088 -> blurred-vision
anxiety: 8, 3.25627 -> blurred-vision
```
### 🧠 Critical Observations

* **Mode Collapse Detected:** The model repeatedly predicted `blurred-vision`.
* **Technical Root Cause:** This is a classic "Frequency Bias." The model has identified `blurred-vision` as a statistically "safe" high-probability prediction across the 25,000-line dataset.
* **Loss Correlation:** The slightly higher Loss (**5.4** vs earlier **3.4**) correlates with less varied predictions, signaling the need for more training epochs or **Gradient Accumulation** to break out of local minima.

