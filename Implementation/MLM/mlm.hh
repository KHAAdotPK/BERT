/*
    mlm/mlm.hh
    Q@hackers.pk    
 */

#ifndef KHAA_PK_BERT_MLM_HH
#define KHAA_PK_BERT_MLM_HH

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 class MLM
 {
    Collective<E> W_mlm; // Projection weights: [d_model x vocab_size]
    Collective<E> b_mlm; // Bias vector: [vocab_size]

    public:
        MLM();
        ~MLM();
        void train(Collective<F>&, Collective<F>&, Collective<F>&, Collective<E>&);    
 };

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::MLM()
 {
    
 }

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::~MLM()
 {
    
 }  

 /*
  * MLM Training Function
  * @param original: Original Input Sequence
  * @param input: Input Sequence with Masked Tokens
  * @param label: Label Sequence
  * @param eo: Encoder Output Sequence   
  *
  * Inside train function, we wil implement the "Project and Compare" logic.
  * Here is a conceptual breakdown of what that function needs to do: 
  * 1. Linear Projection: For every token in the sequence, calculate: eo[i] x W_mlm + b_mlm 
  *    $$\text{logits} = (\text{eo} \cdot W_{mlm}) + b_{mlm}$$
  * 2. Softmax (Optional but recommended): Turn those logits into probabilities.
  *    $$\text{probs} = \text{softmax}(\text{logits})$$
  * 3. Cross-Entropy Loss: Compare the predicted probabilities with the true labels.
  *    $$\text{loss} = -\sum_{i=1}^{n} \text{label}_i \log(\text{probs}_i)$$
  *    - Iterate through your `label` array.
  *    - If label[i] == -100, skip it.
  *    - If label[i] is a valid ID, calculate the difference between the predicted probability and the ground truth.
  *    - Sum up all the losses.
  * 4. Backpropagation: Update $W_{mlm}$ and $b_{mlm}$ using the gradient. 
 */
template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
void MLM<E, F>::train(Collective<F>& original, Collective<F>& input, Collective<F>& label, Collective<E>& eo)  
{
    
}
#endif // KHAA_PK_BERT_MLM_HH