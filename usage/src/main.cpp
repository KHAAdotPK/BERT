/*
    src/main.cpp
    Q@hackers.pk
 */

#include "main.hh"

int main(int argc, char* argv[])
{
    cc_tokenizer::string_character_traits<char>::size_type* ptr = NULL;

    cc_tokenizer::String<char> data; 
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> parser;
    cc_tokenizer::String<char> w1_file_name;
    CORPUS vocab;
    Collective<double> W1;

    // --- Indexing & Probability Pools ---
    // Pool of available positions (0 to mntpl-1) used to select unique token indices for masking.
    cc_tokenizer::string_character_traits<char>::int_type* indices = NULL; 
    // 80/10/10 probability selector: [80: MASK, 90: RANDOM, 100: KEEP]. Shuffled per mutation.
    cc_tokenizer::string_character_traits<char>::int_type* choice_array = NULL; 
    // Shuffled pool of all valid Vocabulary IDs used to provide "Random Damage" tokens.
    cc_tokenizer::string_character_traits<char>::int_type* vocab_indices = NULL; 

    // --- Training Tensors (The Triple-Buffer) ---
    // The 'Ground Truth': Immutable copy of original vocabulary IDs for the current sequence.
    Collective<cc_tokenizer::string_character_traits<char>::int_type> original;
    // The 'Encoder Input': Contains [MASK] IDs, Random Swaps, or Original IDs for the model to process.     
    Collective<cc_tokenizer::string_character_traits<char>::int_type> input;
    // The 'Loss Target': Stores original IDs for mutated tokens; set to -100 for all context/padding tokens.
    Collective<cc_tokenizer::string_character_traits<char>::int_type> label;

    // --- Metadata & Sequence Counters ---
    // Maximum Number of Tokens Per Line (Total capacity of the buffer/tensor).
    cc_tokenizer::string_character_traits<char>::size_type mntpl; 
    // Number of Tokens Per Line (Actual length of the current sequence before padding).
    cc_tokenizer::string_character_traits<char>::size_type ntpl; 
    // Number of Masked Tokens Per Line (The 15% quota target for the current sequence).
    cc_tokenizer::string_character_traits<char>::size_type nmtpl;
    
    // --- Model & Training State ---
    // The 'Encoder Output': A 3D tensor (Batch x SeqLen x Features) holding the contextual embeddings.
    // This is the "frozen" state from the pre-trained encoder that we feed into the MLM Head.
    Collective<double> eoutput;

    // --- Array Dimensions ---
    DIMENSIONSOFARRAY dimensionsOfArray;
    DIMENSIONS dimensions;

    // To store the name of a file, can be used as a scrtch pad to store any file name
    cc_tokenizer::String<char> filename;
    
    try
    {
        std::cout<< "Reading training data from the file..." << std::endl;
        data = cc_tokenizer::cooked_read<char>(cc_tokenizer::String<char>(TRAINING_DATA_PATH));

        std::cout<< "Creating parser for training data..." << std::endl;
        parser =  cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>(data);

        std::cout<< "Creating vocabulary using the training data parser..." << std::endl;
        vocab = CORPUS(parser);

        std::cout<< "Reading pre-trained word vectors from the file..." << std::endl;
        w1_file_name = cc_tokenizer::String<char>(W1_PATH);
    
        std::cout<< "Creating original weights from the pre-trained word vectors..." << std::endl;
        W1 = Collective<double>(NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});    
        READ_W_BIN_NEW_ONE(W1, w1_file_name, double);

        std::cout<< "Determining maximum sequence length..." << std::endl;
        mntpl = parser.max_sequence_length(); // Maximum number of tokens per line
        parser.reset(LINES);

        // Dimensions of the encoder output for one line of the training data
        ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(3);
        ptr[0] = 1; // Batch size
        ptr[1] = 7; // Number of sequences per line
        ptr[2] = SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; // Embedding vector size, number of features per sequence
        dimensionsOfArray = DIMENSIONSOFARRAY(ptr, 3);
        dimensions = DIMENSIONS(dimensionsOfArray);
        ptr = (cc_tokenizer::string_character_traits<char>::size_type*) cc_tokenizer::allocator<double>().allocate(dimensions.getN());
        eoutput = Collective<double>((double*)ptr, dimensions);

        // File name of the encoder output
        filename = cc_tokenizer::String<char>(ENCODER_OUTPUT_PATH);
                    
        indices = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().allocate(mntpl);
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < mntpl; i++)
        {
            indices[i] = i;
        }

        vocab_indices = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().allocate(vocab.numberOfUniqueTokens());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < vocab.numberOfUniqueTokens(); i++)
        {
            vocab_indices[i] = i + INDEX_ORIGINATES_AT_VALUE;
        }
        
        ptr =  reinterpret_cast<cc_tokenizer::string_character_traits<char>::size_type*>( cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().allocate(mntpl) );
        original = Collective<cc_tokenizer::string_character_traits<char>::int_type>(reinterpret_cast<cc_tokenizer::string_character_traits<char>::int_type*>(ptr), DIMENSIONS{mntpl, 1, NULL, NULL});
        
        ptr =  reinterpret_cast<cc_tokenizer::string_character_traits<char>::size_type*>( cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().allocate(mntpl) );
        input = Collective<cc_tokenizer::string_character_traits<char>::int_type>(reinterpret_cast<cc_tokenizer::string_character_traits<char>::int_type*>(ptr), DIMENSIONS{mntpl, 1, NULL, NULL});      

        ptr =  reinterpret_cast<cc_tokenizer::string_character_traits<char>::size_type*>( cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().allocate(mntpl) );
        label = Collective<cc_tokenizer::string_character_traits<char>::int_type>(reinterpret_cast<cc_tokenizer::string_character_traits<char>::int_type*>(ptr), DIMENSIONS{mntpl, 1, NULL, NULL});

        /*
            The 80-10-10 Rule Implementation: For each of the lines, pick MLM_PROBABILITY of the tokens. For each "picked" token:
            1. 80% chance: Replace the token ID with your MASK_TOKEN_ID ID.
            2. 10% chance: Replace it with a random word ID from the vocabulary.
            3. 10% chance: Leave it as the original word.
         */
        choice_array = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().allocate(3);
        choice_array[0] = 80;
        choice_array[1] = 90;
        choice_array[2] = 100;

        // Random number generator, used for shuffling the indices, choice_array and selecting random words from the vocabulary for damaging the encoder output
        static std::mt19937 gen(std::random_device{}());
    
        while (parser.go_to_next_line() != cc_tokenizer::string_character_traits<char>::eof())
        { 
            ntpl = parser.get_total_number_of_tokens();                               // Number of tokens per line
            nmtpl = std::ceil(ntpl * MLM_PROBABILITY);                                // Number of masked tokens per line

            for (cc_tokenizer::string_character_traits<char>::size_type i = ntpl; i < mntpl; i++)
            {
                original[i] = 0;
            }          
            {
                int i = 0;
                while (parser.go_to_next_token() != cc_tokenizer::string_character_traits<char>::eof())
                {
                    original[i] = vocab(parser.get_current_token());                    
                    input[i] = original[i];

                    i++;
                }
                parser.reset(TOKENS);
            }
            // Initialize the mask_damage_keep_indices array with 0
            for (cc_tokenizer::string_character_traits<char>::size_type i = ntpl; i < mntpl; i++)
            {
                input[i] = 0;
            }

            // Initialize the label array with IGNORE_INDEX
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < ntpl; i++)
            {        
                label[i] = IGNORE_INDEX;
            }
            for (cc_tokenizer::string_character_traits<char>::size_type i = ntpl; i < mntpl; i++)
            {                
                label[i] = 0;
            }

            // Line by line read of the encoder output
            eoutput.read(filename);
            // Uncomment to debug
            /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < eoutput.getShape().getN(); i++)
            {                
                if (i % eoutput.getShape().getNumberOfColumns() == 0)
                {
                    std::cout<< std::endl;
                }
                std::cout<< eoutput[i] << " ";
            }
            std::cout<< std::endl;*/

            std::shuffle(indices, indices + mntpl, gen); // Shuffling the indices
            std::shuffle(vocab_indices, vocab_indices + vocab.numberOfUniqueTokens(), gen); // Shuffling the vocabulary indices
                         
            /*
                Strategy: Unique Mask Selection via Shuffled Index Pool
                ------------------------------------------------------
                To satisfy the 15% masking quota without redundant selection, we 
                draw from a pre-shuffled pool of available sequence positions. 
                
                By tracking our position in the shuffled pool with 'pool_cursor', 
                we guarantee that each selected mask index is unique for the 
                current line, ensuring the mathematical integrity of the MLM task.
            */
            cc_tokenizer::string_character_traits<char>::size_type pool_cursor = 0;
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < nmtpl; i++) // Number of masked tokens per line
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = pool_cursor; j < mntpl; j++) // Maximum number of tokens per line
                {
                    if (indices[j] < ntpl) // Creating a Mutation Map
                    {   
                        std::cout<< parser.get_token_by_number(indices[j] + 1).c_str() << " ["; 

                        std::shuffle(choice_array, choice_array + 3, gen);

                        if (choice_array[0] == 80)
                        {
                            std::cout<< "MASKED";                                                                   
                            label[indices[j]] = input[indices[j]];
                            input[indices[j]] = MASK_TOKEN_ID;
                        }
                        else if (choice_array[0] == 90)                        
                        {                           
                            std::shuffle(vocab_indices, vocab_indices + vocab.numberOfUniqueTokens(), gen); 
                            std::cout<< "RANDOM/DAMAGE";                            
 
                            label[indices[j]] = input[indices[j]];
                            input[indices[j]] = vocab_indices[0];
                        }
                        else if (choice_array[0] == 100)
                        {                            
                            std::cout<< "KEEP"; 
                                
                            label[indices[j]] = input[indices[j]];
                            input[indices[j]] = KEEP_TOKEN_ID;
                        }

                        std::cout<< "] ";    
                                             
                        pool_cursor = j + 1;
                        break;
                    }
                }
            }
        
            std::cout<< std::endl;
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < original.getShape().getN(); i++)
            {
                printf("%d ", original[i]);
            }
            std::cout<< std::endl;                        
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < input.getShape().getN(); i++)
            {
                printf("%d ", input[i]);
            }            
            std::cout<< std::endl;
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < label.getShape().getN(); i++)
            {
                printf("%d ", label[i]);
            }            
            std::cout<< std::endl;   
        }

        // Garbage collection
        cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().deallocate(indices, mntpl);
        cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().deallocate(choice_array, 3);
        cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::int_type>().deallocate(vocab_indices, vocab.numberOfUniqueTokens());

        eoutput.close_read();
    }
    catch (ala_exception& e)
    {       
       std::cerr << "main() -> " << e.what() << std::endl;
    }
    catch (std::length_error& e)
    {
        std::cerr << "main() Error: " << e.what() << std::endl;
    }
    catch (std::bad_alloc& e)
    {
        std::cerr << "main() Error: " << e.what() << std::endl;
    }
    
    return 0;
}