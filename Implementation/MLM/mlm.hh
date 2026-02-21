/*
    mlm/mlm.hh
    Q@hackers.pk    
 */

#ifndef KHAA_PK_BERT_MLM_HH
#define KHAA_PK_BERT_MLM_HH

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 class MLM
 {
    cc_tokenizer::string_character_traits<char>::size_type gradient_accumulation_steps_counter;
  
    /*
        Layer 1.

        Hidden_1 Layer (Set of internal features)
        -----------------------------------------
        With the Hidden_1 Layer, we are going to implement: "Feed Forward Network (FFN)"
        $$Hidden_1 = \text{Activation}(Input \cdot W_{hidden_1} + b_{hidden_1})$$                
     */ 
    Collective<E> w_hidden_1; // Hidden layer weights: [d_model x d_model]
    Collective<E> b_hidden_1; // Hidden layer bias: [1 x d_model]
    /*
        Layer 2.

        Hidden_2 Layer (Set of internal features)
        -----------------------------------------
        With the Hidden_2 Layer, we are going to implement: "Feed Forward Network (FFN)"
        $$Hidden_2 = \text{Activation}(Hidden_1 \cdot W_{hidden_2} + b_{hidden_2})$$        
     */ 
    Collective<E> w_hidden_2; // Hidden layer weights: [d_model x d_model]
    Collective<E> b_hidden_2; // Hidden layer bias: [1 x d_model]
    /*
        Layer 3.

        Output Layer
        ------------
        $$Logits = \text{Activation}(Hidden_2 \cdot W_{output} + b_{output})$$  
    */
    Collective<E> w_output; // Output layer weights: [d_model x vocab_size]
    Collective<E> b_output; // Output layer bias: [1 x vocab_size]

    /*
        Gradient Accumulation (accumulators)
        ------------------------------------
        Instead of updating the weights after every single sentence (batch size = 1), we can accumulate the gradients from multiple sentences (e.g., 16) and then update the weights once. This simulates a larger batch size without requiring more memory.
        Why this helps: With a batch size of 1, the model sees only one example at a time. If that example is an outlier or noisy, the model's weights can shift dramatically in the wrong direction. Accumulating gradients smooths out these updates, leading to more stable training and better generalization.
        Implementation: We can add a counter to your train loop. For every sentence, we calculate the gradients but don't update the weights immediately. We add them to dLogits_dw and dLogits_db. Every 16 iterations, we perform the weight update and then reset the accumulators.
     */
    //* One set of accumulators per layer *//
    Collective<E> dHidden_dw_1; // Gradient of hidden layer 1 with respect to weights. Accumulated gradient: d_Loss/d_w_hidden_1
    Collective<E> dHidden_db_1; // Gradient of hidden layer 1 with respect to bias. Accumulated gradient: d_Loss/d_b_hidden_1

    //* One set of accumulators per layer *//
    Collective<E> dHidden_dw_2; // Gradient of hidden layer 2 with respect to weights. Accumulated gradient: d_Loss/d_w_hidden_2
    Collective<E> dHidden_db_2; // Gradient of hidden layer 2 with respect to bias. Accumulated gradient: d_Loss/d_b_hidden_2

    //* One set of accumulators per layer *//
    Collective<E> dOutput_dw; // Gradient of output layer with respect to weights. Accumulated gradient: d_Loss/d_w_output
    Collective<E> dOutput_db; // Gradient of output layer with respect to bias. Accumulated gradient: d_Loss/d_b_output

    /*
        Forward pass cache. A.k.a. Backward pass Cache
        ----------------------------------------------
        EVERY intermediate result must be cached during forward pass
        because backprop needs them to compute gradients via chain rule.
        
        Rule of thumb: 1 hidden layer = 2 cached tensors (raw + activated)
        You have 2 hidden layers = 4 cached tensors needed.
    
        When we calculate the gradients (Backpropagation), we need to know what the value of hidden was before and after the ReLU. 
        But inference only returns the final logits. If you only return the final logits, the training function has no "memory" of what happened inside the hidden layer.

        Inference: We only need the final result (Logits).
        Training: We need the intermediate results (Hidden Layer) to calculate the gradients.

        By caching last_hidden_raw_1 and last_hidden_activated_1 as class members (this->), we successfully solved the "Training Trap"
        Now, when we transition to the train method, the model will "remember" exactly which neurons were fired during the forward pass,
        allowing the Chain Rule to work its magic.
     */

    // Forward Pass Layer 1 Output Cached For Backward Pass
    Collective<E> last_hidden_raw_1; // Z1 = Input * W_hidden_1 + b_hidden_1 , [mntpl x d_model] * [d_model x d_model] = [mntpl x d_model], [mntpl x d_model] + [1 x d_model] = [mntpl x d_model]                     
    Collective<E> last_hidden_activated_1; // H1 = ReLU(Z1) , (linear transformation) [mntpl x d_model]

    // Forward Pass Layer 2 Output Cached for Backward Pass 
    Collective<E> last_hidden_raw_2; // Z2 = H1 * W_hidden_2 + b_hidden_2 , [mntpl x d_model] * [d_model x d_model] = [mntpl x d_model] , [mntpl x d_model] + [1 x d_model] = [mntpl x d_model]
    Collective<E> last_hidden_activated_2; // H2 = ReLU(Z2) , (linear transformation) [mntpl x d_model]

    Collective<E> forward_propagation(Collective<E>&) throw (ala_exception);
    Collective<E> backward_propagation_old(Collective<E>&, Collective<E>&) throw (ala_exception);
    Collective<E> backward_propagation(Collective<E>&, Collective<E>&) throw (ala_exception);   

    public:
        MLM();
        MLM(CORPUS&, cc_tokenizer::string_character_traits<char>::size_type = SKIP_GRAM_EMBEDDNG_VECTOR_SIZE);
        
        ~MLM();
        Collective<E> infer(Collective<E>&) throw (ala_exception);
        E train(Collective<F>&, Collective<F>&, Collective<F>&, Collective<E>&, E = DEFAULT_LEARNING_RATE) throw (ala_exception);
        E train_old(Collective<F>&, Collective<F>&, Collective<F>&, Collective<E>&, E = DEFAULT_LEARNING_RATE);                
 };

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::MLM() : w_hidden_1(), b_hidden_1(), 
                    w_hidden_2(), b_hidden_2(), 
                    w_output(), b_output(), 
                    dHidden_dw_1(), dHidden_db_1(),
                    dHidden_dw_2(), dHidden_db_2(), 
                    dOutput_dw(), dOutput_db(), 
                    gradient_accumulation_steps_counter(0)
{    
}

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::MLM(CORPUS& vocab, cc_tokenizer::string_character_traits<char>::size_type d_model)
 {     
    // Initialize weights, biases and gradient accumulators
    // ----------------------------------------------------

    /*
        In a Neural Network, the range of your initial weights is critical.
        If they are too large, your gradients will explode; if they are too small, the model will never learn (vanishing gradients).

        In modern deep learning, specific initialization strategies based on the size of the layers are used.
        For a "Head" projecting from d_model (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE) to vocab_size (vocab.numberOfUniqueTokens()), we usually use Xavier (Glorot) Initialization.
        For a model of this size, the weights shoul be close to the theoretical Xavier range (${\pm 0.40}$) and  this is our "Goldilocks" zone.

        Xavier adjacent or xavier adjacent initialization: A widely used method in deep learning to initialize the weights of a neural network to prevent vanishing or exploding gradients. It is also known as Glorot initialization. 
     */

    // Layer 1, weights and biases
    // [d_model x d_model]
    w_hidden_1 = Numcy::Random::randn_xavier<E>(DIMENSIONS{d_model /*Columns*/, d_model /*Rows*/, NULL, NULL}, false);
    // [1 x d_model]
    b_hidden_1 = Numcy::zeros<E>(DIMENSIONS{d_model /*Columns*/, 1 /*Rows*/, NULL, NULL});

    // Layer 2, weights and biases
    // [d_model x d_model]
    w_hidden_2 = Numcy::Random::randn_xavier<E>(DIMENSIONS{d_model /*Columns*/, d_model /*Rows*/, NULL, NULL}, false);
    // [1 x d_model]
    b_hidden_2 = Numcy::zeros<E>(DIMENSIONS{d_model /*Columns*/, 1 /*Rows*/, NULL, NULL});

    // Layer 3, weights and biases
    // [d_model x vocab_size]
    w_output = Numcy::Random::randn_xavier<E>(DIMENSIONS{vocab.numberOfUniqueTokens() /*Columns*/, d_model /*Rows*/, NULL, NULL}, false);
    // [1 x vocab_size]
    b_output = Numcy::zeros<E>(DIMENSIONS{vocab.numberOfUniqueTokens() /*Columns*/, 1 /*Rows*/, NULL, NULL});

    /*
        Gradient Accumulation
        ---------------------
        Instead of updating the weights after every single sentence (batch size = 1), we can accumulate the gradients from multiple sentences (e.g., 16) and then update the weights once. This simulates a larger batch size without requiring more memory.
        Why this helps: With a batch size of 1, the model sees only one example at a time. If that example is an outlier or noisy, the model's weights can shift dramatically in the wrong direction. Accumulating gradients smooths out these updates, leading to more stable training and better generalization.
        Implementation: We can add a counter to your train loop. For every sentence, we calculate the gradients but don't update the weights immediately. We add them to dLogits_dw and dLogits_db. Every 16 iterations, we perform the weight update and then reset the accumulators.        
     */
    dHidden_dw_1 = Numcy::zeros<E>(w_hidden_1.getShape()); // Gradient of hidden layer with respect to weights
    dHidden_db_1 = Numcy::zeros<E>(b_hidden_1.getShape()); // Gradient of hidden layer with respect to bias

    dHidden_dw_2 = Numcy::zeros<E>(w_hidden_2.getShape()); // Gradient of hidden layer with respect to weights
    dHidden_db_2 = Numcy::zeros<E>(b_hidden_2.getShape()); // Gradient of hidden layer with respect to bias

    dOutput_dw = Numcy::zeros<E>(w_output.getShape()); // Gradient of output layer with respect to weights
    dOutput_db = Numcy::zeros<E>(b_output.getShape()); // Gradient of output layer with respect to bias

    gradient_accumulation_steps_counter = 0;
 }

 template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
 MLM<E, F>::~MLM()
 {    
 }

// Forward pass is done, now we need to backpropagate/compute the error/gradient through all n layers.
template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
Collective<E> MLM<E, F>::backward_propagation(Collective<E>& eo, /*The original input embedding [mntpl x d_model]*/ Collective<E>& dLogits /*The error from softmax/loss [mntpl x vocab_size]*/) throw (ala_exception) 
{    
    try
    {
        // ===================================================================================================
        // STEP 1: COMPUTE OUTPUT LAYER GRADIENTS WITH RESPECT TO WEIGHTS (dOutput_dw) AND BIASES (dOutput_db)
        // dLogits [mntpl x vocab_size], this->last_hidden_activated_2 is ($$H_2$$) [mntpl x d_model]
        // We need transpose of ($$H_2$$) to get ($$H_2^T$$) [d_model x mntpl]
        // ====================================================================================================
        // Chain rule: dL/dW_output = dL/dLogits · dLogits/dW_output (gradient of the loss with respect to the output layer weights).
        // Gradient:   dL/dW_output = $$H_{2}^{T}$$ · dLogits
        // Chain rule: dL/db_output = dL/dLogits · dLogits/db_output (gradient of the loss with respect to the output layer biases).
        // Gradient:   dL/db_output = sum(dLogits)
        // Why sum(dLogits)? 
        //  - Since b_output is broadcast across all tokens (same bias for each token),
        //  - the gradient sums over the token dimension (rows).        
        // [mntpl x d_model] -> [d_model x mntpl] 
        Collective<E> h2_transposed = Numcy::transpose(this->last_hidden_activated_2);
        // [d_model x mntpl] · [mntpl x vocab_size] = [d_model x vocab_size]         
        Collective<E> h2_transposed_dot_dLogits = Numcy::dot(h2_transposed, dLogits); // Gradient: dL/dW_output = $$H_{2}^{T}$$ · dLogits           
        // [mntpl x vocab_size] -> [1 x vocab_size]
        Collective<E> dLogits_sum = Numcy::sum(dLogits, AXIS_COLUMN); // Gradient: dL/db_output = sum(dLogits)        
        this->dOutput_dw = this->dOutput_dw + h2_transposed_dot_dLogits;
        this->dOutput_db = this->dOutput_db + dLogits_sum;

        // ================================================================
        // STEP 2: PROPAGATE ERROR BACK THROUGH OUTPUT WEIGHTS TO LAYER 2
        // ================================================================
        // Chain rule: dL/dH2 = dL/dLogits · $$(dLogits/dH2)^T$$
        // Gradient:   dL/dH2 = dLogits · $$(W_output)^T$$ 
        // Why transpose?
        // - Because in the forward pass, we computed H₂ · W_output,
        // - which means each element of logits $$Z_{3}$$ depends on a row of H₂ and a column of W_output.
        // - Going backwards reverses this relationship.
        // dL/dH2 [mntpl x d_model]
        // dLogits [mntpl x vocab_size], this->w_output [d_model x vocab_size], this->w_output $$^T$$ [vocab_size x d_model]
        // [mntpl x vocab_size] * [vocab_size x d_model] = [mntpl x d_model]
        Collective<E> w_output_transposed = Numcy::transpose(this->w_output);
        Collective<E> dH2 = Numcy::dot(dLogits, w_output_transposed);

        // =============================================================
        // STEP 3: LAYER 2 ReLU DERIVATIVE GATE 
        // =============================================================
        // Chain Rule: dL/dZ2 = dL/dH2 · dH2/dZ2
        // Gradient:   dL/dZ2 = dH2 · $$(dH2/dZ2)^T$$ 
        // Gradient:   ∂L/∂Z₂ = ∂L/∂H₂ $$\\odot$$ ReLU'($$Z_{2}$$) // To show that ReLU'($$Z_{2}$$) is same as $$(dH2/dZ2)^T$$. Read about the Jacobian matrix. or the diagonal matrix. 
        // Why ReLU'? (Gate gradient: only flows where Z2 was positive)...
        // - ReLU is piecewise linear, so its derivative is 1 for positive values and 0 for negative values.
        // - This means that only the positive values of $$Z_{2}$$ contribute to the gradient.     
        // =============================================================
        dH2 = Numcy::ReLU_Derivative<E>(dH2, this->last_hidden_raw_2);
        //for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dH2.getShape().getN(); i++)
        //{
        //    if (this->last_hidden_raw_2[i] /* $$Z_2$$[i] */ <= 0)
        //    {
        //        dH2[i] = 0;  // Neuron was OFF - kill gradient
        //    }
            // else: neuron was ON - gradient passes through unchanged  
        //}

        // ========================================================================================
        // STEP 4: COMPUTE LAYER 2 GRADIENTS WITH RESPECT TO WEIGHTS (dHidden) AND BIASES (bHidden)
        // ========================================================================================
        // At this point (after step 3), dH2 has been gated by the $$ReLU$$ derivative gate (Step 3), so it now
        // represents $$\partial L/\partial Z_2$$ (gradient of the loss w.r.t. $$Z_2$$ pre-activation values of layer 2).
        //      In the forward pass (Layer 2), we computed:
        //          $$Z_{2}$$ = $$H_{1}$$ · W_hidden_2 + b_hidden_2  (linear transformation a.k.a pre-activation values) 
        //          $$H_{2}$$ = ReLU($$Z_{2}$$)                      (post-activation values)
        //          In Step 2, we computed ∂L/∂H₂ (error signal from output layer).
        //              Chain Rule: dL/dH2 = dL/dZ2 · dZ2/dH2
        //              Gradient:   dL/dH2 = dLogits · $$(W_output)^T$$ 
        // Now we compute gradients for W_hidden_2 and b_hidden_2.
        // --- Weight Gradient ---
        // Chain rule: ∂L/∂W_hidden_2 = ∂L/∂Z₂ · ∂Z₂/∂W_hidden_2
        // Gradient:   dL/dW_hidden_2 = ∂L/∂Z₂ . $$H_1^T$$ 
        /*
            Derivative Rule for Matrix Multiplication
            -----------------------------------------
            For `Y = X · W`, the gradient is: ∂L/∂W = Xᵀ · ∂L/∂Y
            Applying to our case: Z₂ = H₁ · W_hidden_2
            Therefore: ∂L/∂W_hidden_2 = H₁ᵀ · ∂L/∂Z₂            
            `∂L/∂Z₂` is known (this is `dH2` after ReLU gating) — shape: `[mntpl × d_model]`
         */
        // --- Bias Gradient ---
        // Full chain rule: ∂L/∂b_hidden_2 = ∂L/∂Z₂ · ∂Z₂/∂b_hidden_2
        // Gradient: ∂L/∂b_hidden_2 = sum(∂L/∂Z₂, axis=columns)
        // From Z₂ = H₁ · W_hidden_2 + b_hidden_2
        // Why sum? Because each element of $$b_hidden_2$$ is added to all elements in its column in $$Z_2$$.
        // So, the gradient for each bias element is the sum of the gradients from all samples in the batch.
        // Since b_hidden_2 is broadcast (added to every row of Z₂), we get:
        // ∂Z₂/∂b_hidden_2 = 1 (for each row)
        // Therefore:
        // ∂L/∂b_hidden_2 = sum(∂L/∂Z₂, axis=rows)
        // Dimensions:
        //   H₁ᵀ:         [d_model x mntpl]
        //   dH2 (gated): [mntpl x d_model]
        //   Result:      [d_model x d_model]
        Collective<E> h1_transposed = Numcy::transpose(this->last_hidden_activated_1);
        Collective<E> h1_transposed_dot_dH2 = Numcy::dot(h1_transposed, dH2);
        Collective<E> dH2_sum = Numcy::sum(dH2, AXIS_COLUMN);
        this->dHidden_dw_2 = this->dHidden_dw_2 + h1_transposed_dot_dH2;
        this->dHidden_db_2 = this->dHidden_db_2 + dH2_sum;
        
        // ================================================================
        // STEP 5: PROPAGATE ERROR BACK THROUGH HIDDEN_2 WEIGHTS 
        // ================================================================
        // [d_model x d_model] $$^T$$ = [d_model x d_model]
        Collective<E> w_hidden_2_transposed = Numcy::transpose(this->w_hidden_2);
        // dH2 [mntpl x d_model]
        // w_hidden_2_transposed [d_model x d_model]
        // [mntpl x d_model] * [d_model x d_model] = [mntpl x d_model]
        Collective<E> dH2_dot_w_hidden_2_transposed = Numcy::dot(dH2, w_hidden_2_transposed);

        // ================================================================
        // STEP 6: LAYER 1 ReLU DERIVATIVE GATE
        // Same logic as before, now uses Z1 cache
        // ================================================================
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dH2_dot_w_hidden_2_transposed.getShape().getN(); i++)
        {
            if (this->last_hidden_raw_1[i] <= 0)
            {
                dH2_dot_w_hidden_2_transposed[i] = 0;  // Neuron was OFF - kill gradient
            }
            // else: neuron was ON - gradient passes through unchanged
        }

        // ================================================================
        // STEP 7: LAYER 1 WEIGHT GRADIENTS
        // Same as before, but using dH1 instead of dH2
        // ================================================================
        Collective<E> eo_transposed = Numcy::transpose(eo);
        Collective<E> eo_transpose_dot_dH2_dot_w_hidden_2_transposed = Numcy::dot(eo_transposed, dH2_dot_w_hidden_2_transposed);
        Collective<E> dH2_dot_w_hidden_2_transposed_sum = Numcy::sum(dH2_dot_w_hidden_2_transposed, AXIS_COLUMN);
        this->dHidden_dw_1 = this->dHidden_dw_1 + eo_transpose_dot_dH2_dot_w_hidden_2_transposed;
        this->dHidden_db_1 = this->dHidden_db_1 + dH2_dot_w_hidden_2_transposed_sum;                
    }
    catch (const std::exception& e)
    {
        cc_tokenizer::String<char> message = cc_tokenizer::String<char>("MLM::backward_propagation(Collective<E>&, Collective<E>&) -> ") + cc_tokenizer::String<char>(e.what());
        throw ala_exception(message); 
    }

   return Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}; 
}

template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
Collective<E> MLM<E, F>::backward_propagation_old(Collective<E>& eo, /*The original input embedding [mntpl x d_model]*/ Collective<E>& dLogits /*The error from softmax/loss [mntpl x vocab_size]*/) throw (ala_exception) 
{
    Collective<E> h1_transposed;
    Collective<E> h1_transposed_dot_dLogits;
    Collective<E> dLogits_sum;
    Collective<E> w_output_transposed;
    Collective<E> dLogits_dot_w_output_transposed;
    Collective<E> eo_transposed;
    Collective<E> eo_transpose_dot_dLogits_dot_w_output_transposed;
    Collective<E> dLogits_dot_w_output_transposed_sum;

    try
    {
        //std::cout<< "last_hidden_activated_1 = " << this->last_hidden_activated_1.getShape().getNumberOfRows() << ", " << this->last_hidden_activated_1.getShape().getNumberOfColumns() << std::endl;
        //std::cout<< "dLogits = " << dLogits.getShape().getNumberOfRows() << ", " << dLogits.getShape().getNumberOfColumns() << std::endl;

        // --- STEP 1: OUTPUT LAYER GRADIENTS ---
        // dLogits [mntpl x vocab_size], this->last_hidden_activated_1 = ($$h_1$$) [mntpl x d_model]
        // We need transpose of ($$h_1$$) to get ($$h_1^T$$) [d_model x mntpl]
        h1_transposed = Numcy::transpose(this->last_hidden_activated_1);
        //std::cout<< "h1_transposed = " << h1_transposed.getShape().getNumberOfRows() << ", " << h1_transposed.getShape().getNumberOfColumns() << std::endl;
        // We calculate how the weights between Hidden and Output should change
        // [d_model x vocab_size]
        h1_transposed_dot_dLogits = Numcy::dot(h1_transposed, dLogits);
        //std::cout<< "h1_transposed_dot_dLogits = " << h1_transposed_dot_dLogits.getShape().getNumberOfRows() << ", " << h1_transposed_dot_dLogits.getShape().getNumberOfColumns() << std::endl;

        // Accumulate into our class level storage
        // dLogits_sum [1 x vocab_size], dLogits [mntpl x vocab_size]
        dLogits_sum = Numcy::sum(dLogits, AXIS_COLUMN);
        // dOutPut_dw [d_model x vocab_size], h1_transposed_dot_dLogits [d_model x vocab_size]
        this->dOutput_dw = this->dOutput_dw + h1_transposed_dot_dLogits;
        this->dOutput_db = this->dOutput_db + dLogits_sum;

        //std::cout<< "dLogits_sum = " << dLogits_sum.getShape().getNumberOfRows() << ", " << dLogits_sum.getShape().getNumberOfColumns() << std::endl;        
        //std::cout<< "this->dOutput_db = " << this->dOutput_db.getShape().getNumberOfRows() << ", " << this->dOutput_db.getShape().getNumberOfColumns() << std::endl;

        // --- STEP 2: PROPAGATE ERROR BACK TO HIDDEN ---
        // We move the error 'back' through the output weights.
        // w_output_transposed [vocab_size x d_model]
        w_output_transposed = Numcy::transpose(this->w_output);
        // [mntpl x d_model]
        dLogits_dot_w_output_transposed = Numcy::dot(dLogits, w_output_transposed); // [mntpl x vocab_size] * [vocab_size x d_model] = [mntpl x d_model]

        //std::cout<< "w_output_transposed = " << w_output_transposed.getShape().getNumberOfRows() << ", " << w_output_transposed.getShape().getNumberOfColumns() << std::endl;
        //std::cout<< "dLogits_dot_w_output_transposed = " << dLogits_dot_w_output_transposed.getShape().getNumberOfRows() << ", " << dLogits_dot_w_output_transposed.getShape().getNumberOfColumns() << std::endl;

        // --- STEP 3: APPLY ReLU DERIVATIVE ---
        // The gradient only flows through neurons that were "ON" ( > 0 )
        // We use 'last_hidden_raw_1' (Z1) to check the state.
        /*
            When last_hidden_raw_1[i] <= 0: The ReLU derivative is 0, so you zero out the gradient.
            When last_hidden_raw_1[i] > 0: The ReLU derivative is 1, so you keep the gradient (implicit multiplication by 1).
         */
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dLogits_dot_w_output_transposed.getShape().getN(); i++)
        {
            if (this->last_hidden_raw_1[i] <= 0)
            {
                dLogits_dot_w_output_transposed[i] = 0;
            }
        }

        // --- STEP 4: HIDDEN LAYER GRADIENTS ---
        // Finally, calculate how the weights between Input and Hidden should change.
        // [d_model x mntpl]
        eo_transposed = Numcy::transpose(eo);
        // [d_model x mntpl] * [mntpl x d_model] = [d_model x d_model]
        eo_transpose_dot_dLogits_dot_w_output_transposed = Numcy::dot(eo_transposed, dLogits_dot_w_output_transposed);

        // Accumulate into class level storage
        this->dHidden_dw_1 = this->dHidden_dw_1 + eo_transpose_dot_dLogits_dot_w_output_transposed;
        dLogits_dot_w_output_transposed_sum = Numcy::sum(dLogits_dot_w_output_transposed, AXIS_COLUMN);
        this->dHidden_db_1 = this->dHidden_db_1 + dLogits_dot_w_output_transposed_sum;
    }
    catch (ala_exception& e)
    {
        cc_tokenizer::String<char> message = cc_tokenizer::String<char>("MLM::backward_propagation(Collective<E>&, Collective<E>&) -> ") + cc_tokenizer::String<char>(e.what());
        throw ala_exception(message);   
    }

    return Collective<E>{NULL, DIMENSIONS{0 /*Columns*/, 0 /*Rows*/, NULL, NULL}};
}

 /**
 * ======================================================================================
 * NEURAL NETWORK FORWARD ARCHITECTURE
 * ======================================================================================
 * * These methods implement the dual-stage forward propagation of the MLM head. 
 * To support deep learning, we separate the "Logic" from the "Interface":
 * * 1. forward_pass(): The internal engine. It computes the hidden representation, 
 * applies the non-linear ReLU activation, and caches intermediate states 
 * (Zn and Hn) required for the Chain Rule during training (backpropagation).
 * * 2. infer(): The public wrapper. It provides a clean entry point for making 
 * predictions, abstracting away the internal caching and providing error 
 * handling for the higher-level application logic.
 * * Dimensions Flow:
 * $$[Input: mntpl x d_{model}] -> [Hidden: mntpl x d_{model}]_{n} -> [Output: mntpl x vocab_{size}]$$
 * ======================================================================================
 */

/**
 * @brief Performs the Forward Propagation pass through a n-layer Neural Network.
 * This method transforms raw input embeddings into class logits by passing them through 
 * a hidden representation space. It implements the standard Deep Learning "bottleneck" 
 * architecture, which allows the model to learn complex, non-linear relationships 
 * between medical symptoms that a simple linear model cannot capture.
 *
 * @section Math_Logic Mathematical Transformation:
 * 1. Hidden Layer Projection: Zn = (input * W_hidden_n) + b_hidden_n
 * 2. Activation Function:    Hn = ReLU(Zn)  <-- Introduces non-linearity/sparsity
 * 3. Output Layer Projection: Z_output = (Hn * W_output) + b_output
 *
 * @section Cache_Mechanism Gradient Cache:
 * During this pass, 'last_hidden_raw_n' (Zn) and 'last_hidden_activated_n' (Hn) are 
 * cached as class members. This is critical for Backpropagation, as the ReLU 
 * derivative requires Zn, and the Output Layer weight update requires Hn.
 *
 * @tparam E Data type for floating point calculations (double/float).
 * @tparam F Data type for vocabulary indices.
 * @param eo Input Embedding matrix [sequence_length x d_model].
 * @return Collective<E> Logits matrix [sequence_length x vocab_size].
 * @throws ala_exception if matrix dimensions are incompatible for dot products.
 */ 
template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
Collective<E> MLM<E, F>::forward_propagation(Collective<E>& eo) throw (ala_exception) 
{
    /*
        Input Embedding (eo)
        --------------------
        The input `eo` is a matrix of shape `[mntpl, d_model]`.
        - `mntpl`: The number of tokens in the input sequence (e.g., 512 for BERT).
        - `d_model`: The dimension of the embedding vector (e.g., 768 for BERT).

        Think of it like this:
        If you have a sentence "The cat sat on the mat", and `d_model = 4` (for simplicity),
        your input `eo` might look like:
        [
          [0.1, 0.2, 0.3, 0.4],  // "The"
          [0.5, 0.6, 0.7, 0.8],  // "cat"
          [0.9, 1.0, 1.1, 1.2],  // "sat"
          [0.3, 0.4, 0.5, 0.6],  // "on"
          [0.7, 0.8, 0.9, 1.0],  // "the"
          [0.2, 0.3, 0.4, 0.5]   // "mat"
        ]

        Each row is a vector that represents the meaning of that specific word in the context of the sentence.
        The model learns these vectors during training so that words with similar meanings have similar vectors.

        Inference vs. Training
        ----------------------
        Inference: We feed the input through the network once to get the output.
        Training: We feed the input through the network, calculate the error, and then go backward to update the weights.

        The Math Flow:
        1. $Z_1$ (Hidden Linear): Input $\cdot$ $W_{hidden} + B_{hidden}$
        2. $H_1$ (Activation): $ReLU(Z_1)$ — This is the "Brain" part.
        3. $Z_2$ (Output Linear): $H_1 \cdot$ $W_{output} + B_{output}$
     */

    /*
        Forward Propagation/Pass (Inference)
        ------------------------------------
        Linear Projection: For every token in the sequence, calculate: input[i] x w_hidden_1 + b_hidden_1  
     */

    // Layer 1. Input to hidden transformation
    // ---------------------------------------
    // [mntpl x d_model] * [d_model x d_model] = [mntpl x d_model]
    this->last_hidden_raw_1 = Numcy::dot(eo, this->w_hidden_1); // $$Z_{1} = input \cdot w_{hidden_1}$$ // Linear transformation step 1
    // [mntpl x d_model] + [1 x d_model] = [mntpl x d_model]
    this->last_hidden_raw_1 = this->last_hidden_raw_1 + this->b_hidden_1; // $$Z_{1} = Z_{1} + b_{hidden_1}$$ // Linear transformation step 2
    // [mntpl x d_model]  
    this->last_hidden_activated_1 = Numcy::ReLU(this->last_hidden_raw_1); // $$H_{1} = \text{ReLU}(Z_{1})$$ // Activation function

    // Layer 2. Hidden to hidden transformation
    // ----------------------------------------
    // [mntpl x d_model] * [d_model x d_model] = [mntpl x d_model]
    this->last_hidden_raw_2 = Numcy::dot(this->last_hidden_activated_1, this->w_hidden_2); // $$Z_{2} = H_{1} \cdot w_{hidden_2}$$ // Linear transformation step 1
    // [mntpl x d_model] + [1 x d_model] = [mntpl x d_model]
    this->last_hidden_raw_2 = this->last_hidden_raw_2 + this->b_hidden_2; // $$Z_{2} = Z_{2} + b_{hidden_2}$$ // Linear transformation step 2
    // [mntpl x d_model]  
    this->last_hidden_activated_2 = Numcy::ReLU(this->last_hidden_raw_2); // $$H_{2} = \text{ReLU}(Z_{2})$$ // Activation function    
    
    // ACTIVATION (Hidden to Hidden, Non-Linearity)
    /*
        Why we need the ReLU?
        ---------------------
        ReLU is a non-linear activation function that is used to introduce non-linearity into the network. 
        It is a simple activation function that returns the maximum of 0 and the input value. 
        It is a piecewise linear function that is defined as: $f(x) = max(0, x)$

        Without Numcy::relu(),
        two linear layers are mathematically identical to one big linear layer ($W_2 \cdot (W_1 \cdot x) = (W_2 \cdot W_1) \cdot x$).
        The ReLU "breaks" the linearity, allowing the model to learn that symptoms like sneezing only matter if runny-nose is also present.

        more:
        Without an activation function like ReLU, adding ten hidden layers is mathematically the same as having just one layer.
        ReLU introduces non-linearity, which allows the model to learn that runny-nose + fever means something different than just runny-nose alone.

        more:
        ReLU is a simple activation function that returns the maximum of 0 and the input value. 
        With ReLU ($max(0, x)$), the neuron stays at 0 (completely silent) unless the input signals are strong enough.
        This creates Sparsity, which is how BERT manages to be so smart.
     */
 
    // Layer 3. Hidden to output transformation (final linear projection)
    // ------------------------------------------------------------------
    /*
        Output Layer
        ------------
        Linear Projection: For every token in the sequence, calculate: hidden[i,j] x w_output[j,i] + b_output[j]  
    */
    // [mntpl x d_model] * [d_model x vocab_size] = [mntpl x vocab_size]
    // Each of the (mntpl many) tokens now has (vocab_size many) scores (one per word in vocab).
    // Softmax will turn these into probabilities during training and inference (mlm::train(), mlm::infer())
    Collective<E> logits /* $$Z_{3}$$ */ = Numcy::dot(this->last_hidden_activated_2, this->w_output); // $$Z_{3} = H_{2} \cdot w_{output}$$
    // [mntpl x vocab_size] + [1 x vocab_size] = [mntpl x vocab_size]
    logits = logits + this->b_output; // $$Z_{3} = Z_{3} + b_{output}$$
    
    // [mntpl x vocab_size] . Inference: We only need the final result (Logits).
    return logits; // Pass by value, not by reference
}

/*
    Inference: We only need the final result (Logits).
    Training: We need the intermediate results (Hidden Layer) to calculate the gradients.
 */ 
template <typename E = double, typename F = cc_tokenizer::string_character_traits<char>::int_type>
Collective<E> MLM<E, F>::infer(Collective<E>& eo) throw (ala_exception) 
{
    Collective<E> logits;

    try 
    {
       logits = forward_propagation(eo);
    }
    catch (ala_exception& e)
    {        
        cc_tokenizer::String<char> message = cc_tokenizer::String<char>("MLM::infer(Collective<E>&) -> ") + cc_tokenizer::String<char>(e.what());

        throw ala_exception(message);
    }

    return logits;
}

/*
 * Training: We need the intermediate results (Hidden Layer) to calculate the gradients
 * ------------------------------------------------------------------------------------
 * MLM Training Function
 * @param original: Original Input Sequence
 * @param input: Input Sequence with Masked Tokens
 * @param label: Label Sequence
 * @param eo: Encoder Output Sequence   
 *
 * The original and input tokens are usually used in the data-generator phase to create the masks.
 * The label tokens are the original tokens that were replaced with [MASK] tokens.
 * In a Masked Language Model (MLM), once you have the eo (Encoder Output/Hidden States),
 * you technically only need the label (the ground truth) to calculate loss and backpropagate.
 * 
 * The eo tokens are the encoder output tokens that were replaced with [MASK] tokens.
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
E MLM<E, F>::train(Collective<F>& original, Collective<F>& input, Collective<F>& label, Collective<E>& eo, E learning_rate) throw (ala_exception) 
{
    Collective<E> logits, logits_row;
    Collective<E> predicted_probabilities;
    Collective<E> dLogits;
    F idx = 0;

    E loss = 0;
    cc_tokenizer::string_character_traits<char>::size_type mask_count = 0; // Initialize mask count to zero

    // 0. ACCUMULATE GRADIENTS (Summing up gradients from mini-batches)
    // We divide by GRADIENT_ACCUMULATION_STEPS to average them out.
 
    try 
    {
        // 1. FORWARD PASS
        // This fills last_hidden_raw_1 and last_hidden_activated_1 caches
        logits = forward_propagation(eo);

        //std::cout<< "logits = " << logits.getShape().getNumberOfRows() << ", " << logits.getShape().getNumberOfColumns() << std::endl;

        // 2. COMPUTE GRADIENTS AT THE OUTPUT (Softmax + Cross-Entropy Error)
        // dLogits = Probabilities - Targets
        // Numcy::softmax handles the probability distribution

        dLogits = Numcy::zeros<E>(logits.getShape()); 

        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN() && label[i] != INDEX_NOT_FOUND_AT_VALUE; i++)
        {    
            logits_row = logits.slice(i*logits.getShape().getNumberOfColumns(), DIMENSIONS{logits.getShape().getNumberOfColumns() /*Columns*/, 1 /*Rows*/, NULL, NULL});
            /*
                We use Temperature only during inference (testing the model).
                During training, we want the model to learn the true distribution of the data, 
                so we don't use temperature scaling.
                Train at $T=1.0$ (This keeps the "math" natural).
            */
            predicted_probabilities = Numcy::softmax(logits_row, NATURAL_TEMPERATURE); // Probabilities for just one row of logits, each token of the sequence has its own logits row.

            if (label[i] != IGNORE_INDEX)
            {
                if (input[i] == MASK_TOKEN_ID)
                {
                    // MASKED
                    idx = label[i];
                }
                else if (input[i] == label[i] /*KEEP*/ || input[i] != label[i] /*DAMAGED*/) // For documentation purposes only
                {
                    // KEEP OR DAMAGED
                    idx = label[i];
                }

                // Gradient: probability - 1 for correct class, probability - 0 for others
                dLogits[i*dLogits.getShape().getNumberOfColumns() + idx - INDEX_ORIGINATES_AT_VALUE] = predicted_probabilities[idx - INDEX_ORIGINATES_AT_VALUE] - 1;

                // Gradient: probability - 0 for others
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < predicted_probabilities.getShape().getN(); j++)
                {
                    if (j != idx - INDEX_ORIGINATES_AT_VALUE)
                    {
                        dLogits[i*dLogits.getShape().getNumberOfColumns() + j] = predicted_probabilities[j] - 0;
                    }
                }

                loss += -log(predicted_probabilities[idx - INDEX_ORIGINATES_AT_VALUE] + EPSILON);
                mask_count++;
            }
            // NOTE :- DONOT REMOVE FOLLOWING COMMENT BLOCK, KEEP IT FOR DOCUMENTATION PURPOSES
            /*else // IGNORE_INDEX positions, we are not setting gradients for IGNORE_INDEX positions
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < predicted_probabilities.getShape().getN(); j++)
                {
                    dLogits[i*dLogits.getShape().getNumberOfColumns() + j] = predicted_probabilities[j] - 0;
                }
            }*/
                         
            idx = 0;
        }
        if (mask_count > 0)
        {
            loss /= mask_count;
        }

        // 3. BACK PROPAGATION
        /*Collective<E> dEo =*/ this->backward_propagation(eo, dLogits);

        // 4. GRADIENT ACCUMULATION & WEIGHT UPDATE
        this->gradient_accumulation_steps_counter++;

        /*double scale = 1.0 / this->gradient_accumulation_steps_counter;*/

        /*this->dOutput_dw = this->dOutput_dw * scale;
        this->dOutput_db = this->dOutput_db * scale;
        this->dHidden_dw_1 = this->dHidden_dw_1 * scale;
        this->dHidden_db_1 = this->dHidden_db_1 * scale;*/

        // Update Output Layer: W = W - (LR * dW)
        /*this->w_output = this->w_output - (this->dOutput_dw * learning_rate);
        this->b_output = this->b_output - (this->dOutput_db * learning_rate);*/

        // Update Hidden Layer: W = W - (LR * dW)
        /*this->w_hidden_1 = this->w_hidden_1 - (this->dHidden_dw_1 * learning_rate);
        this->b_hidden_1 = this->b_hidden_1 - (this->dHidden_db_1 * learning_rate);*/

        // Only update weights every GRADIENT_ACCUMULATION_STEPS steps
        if (this->gradient_accumulation_steps_counter >= GRADIENT_ACCUMULATION_STEPS)
        {
            double scale = 1.0 / GRADIENT_ACCUMULATION_STEPS;

            // We divide by GRADIENT_ACCUMULATION_STEPS to average them out.
            this->dOutput_dw = this->dOutput_dw * scale;
            this->dOutput_db = this->dOutput_db * scale;
            this->dHidden_dw_2 = this->dHidden_dw_2 * scale;  // NEW
            this->dHidden_db_2 = this->dHidden_db_2 * scale;  // NEW
            this->dHidden_dw_1 = this->dHidden_dw_1 * scale; 
            this->dHidden_db_1 = this->dHidden_db_1 * scale;

            // Update Output Layer: W = W - (LR * dW)
            this->w_output = this->w_output - (this->dOutput_dw * learning_rate);
            this->b_output = this->b_output - (this->dOutput_db * learning_rate);
            this->w_hidden_2 = this->w_hidden_2 - (this->dHidden_dw_2 * learning_rate);  // NEW
            this->b_hidden_2 = this->b_hidden_2 - (this->dHidden_db_2 * learning_rate);  // NEW
            // Update Hidden Layer: W = W - (LR * dW)
            this->w_hidden_1 = this->w_hidden_1 - (this->dHidden_dw_1 * learning_rate);
            this->b_hidden_1 = this->b_hidden_1 - (this->dHidden_db_1 * learning_rate);
            
            this->gradient_accumulation_steps_counter = 0;

            // RESET ACCUMULATORS
            this->dOutput_dw = Numcy::zeros<E>(this->w_output.getShape());
            this->dOutput_db = Numcy::zeros<E>(this->b_output.getShape());
            this->dHidden_dw_2 = Numcy::zeros<E>(this->w_hidden_2.getShape());  // NEW
            this->dHidden_db_2 = Numcy::zeros<E>(this->b_hidden_2.getShape());  // NEW
            this->dHidden_dw_1 = Numcy::zeros<E>(this->w_hidden_1.getShape());
            this->dHidden_db_1 = Numcy::zeros<E>(this->b_hidden_1.getShape());
        }
    }    
    catch (ala_exception& e)
    {        
        cc_tokenizer::String<char> message = cc_tokenizer::String<char>("MLM::train(Collective<F>&, Collective<F>&, Collective<F>&, Collective<E>&, E) -> ") + cc_tokenizer::String<char>(e.what());

        throw ala_exception(message);
    }

    // 5. RETURN LOSS (Optional: for monitoring progress)    
    return loss;
}

#endif // KHAA_PK_BERT_MLM_HH