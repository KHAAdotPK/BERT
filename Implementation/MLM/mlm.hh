/*
    mlm/mlm.hh
    Q@hackers.pk    
 */

#ifndef KHAA_PK_BERT_MLM_HH
#define KHAA_PK_BERT_MLM_HH

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 class MLM
 {
    Collective<E> w_mlm; // Projection weights: [d_model x vocab_size]
    Collective<E> b_mlm; // Bias vector: [vocab_size]

    E loss;

    public:
        MLM();
        MLM(CORPUS&, cc_tokenizer::string_character_traits<char>::size_type = SKIP_GRAM_EMBEDDNG_VECTOR_SIZE);

        ~MLM();
        void train(Collective<F>&, Collective<F>&, Collective<F>&, Collective<E>&);    
 };

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::MLM() : w_mlm(), b_mlm(), loss()
 {
    
 }

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::MLM(CORPUS& vocab, cc_tokenizer::string_character_traits<char>::size_type d_model) : loss()
 {     
    // Initialize weights and biases

    /*
        In a Neural Network, the range of your initial weights is critical.
        If they are too large, your gradients will explode; if they are too small, the model will never learn (vanishing gradients).

        In modern deep learning, specific initialization strategies based on the size of the layers are used.
        For a "Head" projecting from d_model (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE) to vocab_size (vocab.numberOfUniqueTokens()), we usually use Xavier (Glorot) Initialization.
        For a model of this size, the weights shoul be close to the theoretical Xavier range (${\pm 0.40}$) and  this is our "Goldilocks" zone.

        Xavier adjacent or xavier adjacent initialization: A widely used method in deep learning to initialize the weights of a neural network to prevent vanishing or exploding gradients. It is also known as Glorot initialization. 
     */
    w_mlm = Numcy::Random::randn_xavier<E>(DIMENSIONS{vocab.numberOfUniqueTokens(), d_model, NULL, NULL}, false);
    // Initialize bias with zeros
    b_mlm = Numcy::zeros<E>(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
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
void MLM<E, F>::train(Collective<F>& original, Collective<F>& input, Collective<F>& label, Collective<E>& eo) throw (ala_exception) 
{
    /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < w_mlm.getShape().getN(); i++)
    {
        std::cout<< w_mlm[i] << " ";

        if ((i + 1) % w_mlm.getShape().getNumberOfColumns() == 0)
        {
            std::cout<< std::endl;
        }
    }*/

    
    /*
        Step 1 (TODO)
        ------
        Before doing any math, ensure the shapes align.
        - **Input check:** Ensure eo.rows matches the sequence length (maximum number of tokens per line).
        - ""Matrix check:"" Ensure eo.cols matches W_mlm.rows.
        - **Bias check:** Ensure b_mlm size matches the vocabulary size (vocab.numbersOfUniqueTokens()).
     */
    if (!w_mlm.getShape().getN() || !b_mlm.getShape().getN())
    {
        throw ala_exception("MLM::train(Collective<F>&, Collective<F>&, Collective<F>&, Collective<E>&) Error: W_mlm or b_mlm is not initialized");
    }

    /*
        Step 2 (Forwar Propagation) 
        ---------------------------
        - Linear Projection: For every token in the sequence, calculate: eo[i] x W_mlm + b_mlm 
            - Perform the matrix multiplication to compute the Logits.
            - $$\text{logits} = (\text{eo} \cdot W_{mlm}) + b_{mlm}$$
            - You will end up with a $(eo.getShape().getNumberOfRows() \times vocab.getShape().numbersOfUniqueTokens())$ matrix.
            - Each row represents a position in the sentence, and each column is the "score" for a specific word in your dictionary.    
            - The higher the score, the more likely the word is to be the correct word at that position.
            - The lower the score, the less likely the word is to be the correct word at that position.
     */     
    Collective<E> logits = Numcy::dot(eo, w_mlm);
    logits = logits + b_mlm;

    /*std::cout<< "Logits Rows: " << logits.getShape().getNumberOfRows() << " " << logits.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "b_mlm Rows: " << b_mlm.getShape().getNumberOfRows() << " " << b_mlm.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "w_mlm Rows: " << w_mlm.getShape().getNumberOfRows() << " " << w_mlm.getShape().getNumberOfColumns() << std::endl;*/

    /*DIMENSIONSOFARRAY dims = logits.getShape().getDimensionsOfArray();

    std::cout<< "Number of Dimensions: " << dims.size() << std::endl;
    std::cout<< "Dimensions: -> " << dims[0] << " -> " << dims[1] << " -> " << dims[2] << << std::endl;*/

    Collective<E> logits_row;
    Collective<E> predicted_probabilities;
    double loss = 0.0; // Initialize Cross-Entropy loss to zero
    cc_tokenizer::string_character_traits<char>::size_type mask_count = 0; // Initialize mask count to zero
    /*
        Step 3 Logit-to-Label Mapping (Mask Filtering) 
        ----------------------------------------------
        - Iterate through your `label` array.
        - If label[i] == -100, skip it.
        - For every index $i$ where label[i] != -100, extract the corresponding row from your Logits matrix.
        - If label[i] is a valid ID, calculate the difference between the predicted probability and the ground truth.
        - Sum the losses for all masked positions and divide by the number of masks to get the average.
     */ 
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN(); i++)
    {
        if (label[i] != IGNORE_INDEX && label[i] != INDEX_NOT_FOUND_AT_VALUE)
        {
            logits_row = logits.slice(i*logits.getShape().getNumberOfColumns(), DIMENSIONS{logits.getShape().getNumberOfColumns(), 1, NULL, NULL});

            /*std::cout<< "Logits Row: ";
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < logits_row.getShape().getNumberOfColumns(); j++)
            {
                std::cout<< logits_row[j] << " ";
            }
            std::cout<< std::endl;*/

            predicted_probabilities = Numcy::softmax(logits_row);

            /*std::cout<< "predicted " << predicted_probabilities.getShape().getNumberOfRows() << " " << predicted_probabilities.getShape().getNumberOfColumns() << std::endl;
            std::cout<< "Label: " << label[i] << std::endl;*/

            loss += -log(predicted_probabilities[label[i] - INDEX_ORIGINATES_AT_VALUE]);

            mask_count++;            
        }
    }        
    loss /= mask_count;

    /*
        Step 6: Backpropagation (Gradient Calculation)
        ----------------------------------------------
        This is where the learning happens. You calculate how to nudge $W_{mlm}$ and $b_{mlm}$ to make the loss smaller.
        - Calculate the gradient of the loss with respect to the logits.
        - $$\text{dLoss/dLogits} = \text{probs} - \text{labels}$$
        - This tells you how much each logit contributed to the error.
    */ 
    // Prepare a matrix to hold gradients for the entire sequence, I should have named it dLoss_dLogits
    Collective<E> dLogits = Numcy::zeros<E>(logits.getShape()); 
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN(); i++)
    {
        if (label[i] != IGNORE_INDEX && label[i] != INDEX_NOT_FOUND_AT_VALUE)
        {
            //dLogits[i] = predicted_probabilities[i] - (label[i] == i ? 1 : 0);

            // --- STEP 6: GRADIENT FOR THIS ROW ---
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*logits.getShape().getNumberOfColumns()*/ predicted_probabilities.getShape().getN(); j++)
            {
                //dLogits[i * logits.getShape().getNumberOfColumns() + j] = predicted_probabilities[i * logits.getShape().getNumberOfColumns() + j] - (label[i] == j ? 1 : 0);
                E prob = predicted_probabilities[j];
                if (j == label[i] - INDEX_ORIGINATES_AT_VALUE)
                {
                    dLogits[i*logits.getShape().getNumberOfColumns() + j] = prob - 1.0;
                }
                else
                {
                    dLogits[i*logits.getShape().getNumberOfColumns() + j] = prob;
                }
            }
        }
    }
    /*
        Step 6: Gradient Normalization
        ------------------------------
        To make sure the gradients are not too large, we divide them by the number of masked tokens.

        Make sure that mask_count is at least 1 before dividing, 
        or might get a NaN error on lines where no tokens were selected for training (though our noise generator seems to always pick at least one).
     */
    if (mask_count > 0)
    {
        dLogits = dLogits / mask_count;
    }

    /*
        Step 7: Optimization (The Update)
        ---------------------------------
        Once you have dLogits (how much the scores should change), you need to find out how much the weights should change.
        This is the "chain rule" in action.

        $$\text{dLoss/dW} = \text{dLoss/dLogits} \times \text{dLogits/dW}$$
        $$\text{dLoss/db} = \text{dLoss/dLogits} \times \text{dLogits/db}$$

        Where:
        - $\text{dLoss/dLogits}$ is the gradient of the loss with respect to the logits.
        - $\text{dLogits/dW}$ is the gradient of the logits with respect to the weights.
        - $\text{dLogits/db}$ is the gradient of the logits with respect to the bias.

        Collective<E> dLoss_dLogits = dLogits;
        Collective<E> dLogits_dW = Numcy::zeros<E>(w_mlm.getShape());
        Collective<E> dLogits_db = Numcy::zeros<E>(b_mlm.getShape());      
    */
    Collective<E> dLoss_dLogits = dLogits;
    Collective<E> dLogits_dW = Numcy::zeros<E>(w_mlm.getShape());
    Collective<E> dLogits_db = Numcy::zeros<E>(b_mlm.getShape());

    Collective<E> eo_transposed = Numcy::transpose(eo);

    /*std::cout<< "------->>>>>>>>>>>>>> " << eo_transposed.getShape().getDimensionsOfArray().size() << std::endl;
    std::cout<< "Columns eo = " << eo_transposed.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "Rows = " << eo_transposed.getShape().getNumberOfRows() << std::endl;*/

    dLogits_dW = Numcy::dot(eo_transposed, dLogits);
    dLogits_db = Numcy::sum(dLogits, AXIS_COLUMN);

    /*std::cout<< "Columns db = " << dLogits_db.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "Rows = " << dLogits_db.getShape().getNumberOfRows() << std::endl;*/

    double learning_rate = 0.01; // Or whatever alpha you prefer
    /*
        Step 8: The Weight Update (The Finale)
        --------------------------------------
        Now that you have your gradients (dLogits_dW and dLogits_db), we have the direction of the mountain's slope.
        To actually move down the mountain, you need a Learning Rate ($$\eta$$).
        Without it, your update will be too aggressive and will likely "explode" your weights.

        $$\theta_{new} = \theta_{old} - \eta \cdot \frac{\partial J}{\partial \theta}$$

        Where:
        - $$\\theta_{new}$$: The updated weight/bias.
        - $$\\theta_{old}$$: The current weight/bias.
        - $$\\eta$$: The Learning Rate (e.g., 0.001).
        - $$\\frac{\\partial J}{\\partial \\theta}$$: The gradient you just calculated.
     */
    // W = W - (lr * dW)
    // w_mlm = w_mlm - learning_rate * dLogits_dW;
    w_mlm = w_mlm - (dLogits_dW * learning_rate);

    /*std::cout<< "Columns w_mlm = " << w_mlm.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "Rows w_mlm = " << w_mlm.getShape().getNumberOfRows() << std::endl;

    std::cout<< "Columns dLogits_dW = " << dLogits_dW.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "Rows dLogits_dW = " << dLogits_dW.getShape().getNumberOfRows() << std::endl;*/

    // b = b - (lr * db)
    // b_mlm = b_mlm - learning_rate * dLogits_db; 
    b_mlm = b_mlm - (dLogits_db * learning_rate);

    std::cout<< "Loss = " << loss << std::endl;

    /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN(); i++)
    {
        if (label[i] != IGNORE_INDEX && label[i] != INDEX_NOT_FOUND_AT_VALUE)
        {
            // Get the specific row for this token
            Collective<E> row_probs = predicted_probabilities.slice(i * vocab_size, DIMENSIONS{vocab_size, 1, NULL, NULL});
            
            // Get the correct label index
            cc_tokenizer::string_character_traits<char>::size_type correct_idx = label[i] - INDEX_ORIGINATES_AT_VALUE;
            
            // Calculate dLoss/dLogits for this row: (Probs - 1) at correct_idx, (Probs - 0) elsewhere
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < vocab_size; j++)
            {
                E error = row_probs[j];
                if (j == correct_idx)
                {
                    error -= 1.0;
                }
                dLogits[i * vocab_size + j] = error;
            }
        }
    }*/
 

    /*
        Step 
     */

    
    /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN(); i++)
    {
        if (label[i] != IGNORE_INDEX && label[i] != INDEX_NOT_FOUND_AT_VALUE)
        {
            // Get the specific row for this token
            Collective<E> row_probs = predicted_probabilities.slice(i * vocab_size, DIMENSIONS{vocab_size, 1, NULL, NULL});
            
            // Get the correct label index
            cc_tokenizer::string_character_traits<char>::size_type correct_idx = label[i] - INDEX_ORIGINATES_AT_VALUE;
            
            // Calculate dLoss/dLogits for this row: (Probs - 1) at correct_idx, (Probs - 0) elsewhere
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < vocab_size; j++)
            {
                E error = row_probs[j];
                if (j == correct_idx)
                {
                    error -= 1.0;
                }
                dLogits[i * vocab_size + j] = error;
            }
        }
    }*/
    

    /*Collective<E> dLoss_dLogits = predicted_probabilities;
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN(); i++)
    {
        if (label[i] != IGNORE_INDEX && label[i] != INDEX_NOT_FOUND_AT_VALUE)
        {
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < predicted_probabilities.getShape().getN(); j++)
            {               
                dLoss_dLogits[j] -= dLoss_dLogits[label[i] - INDEX_ORIGINATES_AT_VALUE];
            }
        }
    }*/
    
    

      

    /*
        - Softmax: Convert logits to probabilities.
        - $$\text{probs} = \text{softmax}(\text{logits})$$
        - Cross-Entropy Loss: Compare the predicted probabilities with the true labels.
            - $$\text{loss} = -\sum_{i=1}^{n} \text{label}_i \log(\text{probs}_i)$$

        - Backpropagation: Update $W_{mlm}$ and $b_{mlm}$ using the gradient. */     
}
#endif // KHAA_PK_BERT_MLM_HH