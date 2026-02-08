/*
    src/main.hh
    Q@hackers.pk
 */

#include <iostream> // I die each time I write this 

#include "./../../Implementation/lib/csv/parser.hh" 
#include "./../../Implementation/lib/Numcy/header.hh"
#include "./../../Implementation/lib/Corpus/corpus.hh"
#include "./../../Implementation/lib/Sundry/cooked_read_new.hh"
#include "./../../Implementation/lib/read_write_weights/header.hh"
#include "./../../Implementation/lib/argsv-cpp/lib/parser/parser.hh"

#define COMMAND "verbose --verbose (TODO)\n\
infer --infer (TODO)\n\
top-k --top-k (TODO)\n"

#define DEFAULT_INFER_LINE 1
#define DEFAULT_TOP_K 5

#define NATURAL_TEMPERATURE 1.0
#define HIGH_TEMPERATURE 2.0
#define LOW_TEMPERATURE 0.5

/*
    The training process uses a Gradient Accumulation strategy to simulate batch training on a CPU without increased memory overhead. 
    You can tune the training stability and speed using the GRADIENT_ACCUMULATION_STEPS macro.

    The value (default 16) represents how many sentences are "read" and analyzed before the model makes a single change to its weights.
    - The Power of 2: Always try to keep this number as a power of 2 (2, 4, 8, 16, 32...).
      This allows the C++ compiler to optimize the math for your CPU’s cache lines, leading to slightly faster execution.
    - The Learning Rate Connection: If you increase the accumulation steps (e.g., from 16 to 32), the model becomes more "confident" in its direction.
      You can often slightly increase your learning rate to take bigger, more stable steps.
    - Memory Efficiency: Because this is implemented via accumulation in the MLM class, increasing this number does not use more RAM.
      It only changes how often the w_mlm and b_mlm weights are updated. 

   Advice
   ------
   - Start with a small value (e.g., 2) and increase it gradually to find the optimal balance between speed and stability.
   - Monitor the training process closely to ensure that the model is learning effectively.
   - If you notice any instability or divergence, consider decreasing the accumulation steps.

   Stick with 16 for first "Batching" experiment. Once you see the loss curve in your logs, if it looks too flat, move down to 8. If it's still too chaotic, move up to 32.
 */
#define GRADIENT_ACCUMULATION_STEPS 16

#ifndef MLM_BERT_NO_MORE_AUTO_REGRESSIVE_MODEL // In C/C++ world donot travel without include guards; period
#define MLM_BERT_NO_MORE_AUTO_REGRESSIVE_MODEL

#define TRAINING_DATA_PATH "./../data/vocabulary/INPUT.txt"
#define W1_PATH "./../data/weights/w1p.dat"
#define ENCODER_OUTPUT_PATH "./../data/weights/encoder_output.dat"
#define ORIGINAL_INPUT_LABEL_PATH = "./../data/weights/original_input_label.dat"

#define MLM_OUTPUT_PATH "./../data/weights/mlm_output.dat"
#define MLM_LABELS_PATH "./../data/weights/mlm_labels.dat"
#define MLM_WEIGHTS_PATH "./../data/weights/mlm_weights.dat"
#define MLM_BIAS_PATH "./../data/weights/mlm_bias.dat"

/*
    What is MLM?
    ------------
    In BERT implementations, the value 0.15 refers to the probability of a token being selected for the Masked Language Modeling (MLM) task.
    Based on standard libraries like Hugging Face and common academic naming conventions, 
    you should use one of the following macro or constant names: 
    - MLM_PROBABILITY: This is the most common parameter name used in industry-standard libraries (e.g., Hugging Face's mlm_probability).
    - MASKING_RATIO: Widely used in research papers to describe the 15% rate of token selection.
    - MASK_RATE: A concise alternative frequently used in deep learning implementations. 

    Context for 0.15 in BERT
    -------------------------
    In BERT, the value 0.15 is used as the probability of selecting a token for masking during the training process. 
    This means that 15% of the tokens in the input sequence are randomly selected and replaced with a special mask token (e.g., [MASK]). 
    The remaining 85% of the tokens are left unchanged. This masking process is a key component of the BERT training procedure, 
    as it helps the model learn to predict the original tokens from the masked versions, 
    which is a form of self-supervised learning.

    "Damaging" the Data
    -------------------
    In the original BERT architecture, this 15% of selected tokens is further processed using a 80-10-10 rule:
    - 80% are replaced with the [MASK] token. Replace the token ID with your [MASK] ID (e.g., ID 28).
    - 10% are replaced with the same token (i.e., a no change operation, the original token is kept, leave it as the original word). 
    - 10% are replaced with a random token from the vocabulary (replace it with a random word ID from your vocabulary).

    The Output of this pipe
    -----------------------
    - input_ids: The "damaged" sentence for the model to see.
    - labels: A "cheat sheet" containing the original word IDs only at the masked positions (all other positions should be a "ignore" value like -100).
    Why -100? > Most Loss Functions (like Cross-Entropy) in C++ are written to ignore the value -100.
    This tells the model: "Only calculate the 'Pain' for the bracketed word; don't worry about the others."
 */
 #define MLM_PROBABILITY 0.15
 #define MASK_TOKEN_ID -28
 /* 
    Target/Label Array
    ------------------
    For every line, you need a Labels array. 
    The size of the labels array should be equal to the number of tokens in the line.
    The labels array should contain the original word IDs only at the masked positions (all other positions should be a "ignore" value like -100).
    This is a common practice in masked language modeling tasks, where the model is only concerned with the loss at the positions where the tokens have been masked.
    
    Example:
    Input: [1, 2, 3, 4, 5]
    Masked: [KEEP_TOKEN_ID, 2, MASK_TOKEN_ID, 4, RANDOM_TOKEN_ID]
    Labels: [1, IGNORE_INDEX, 3, IGNORE_INDEX, 5]

    The "Why" Behind the Labels
    ---------------------------
    - **KEEP_TOKEN_ID (1):** The model sees the word "cat" and is told to "keep it." to verify that this word is actually correct in this context.
        - **Why?** To teach the model that not all words should be changed. It helps the model learn the context of the surrounding words.
    - **MASK_TOKEN_ID (-28):** The model sees a "masked" word and asks "What word goes here?"
        - **Why?** This is the core of the task. The model must predict what word goes here based on the context.
    - **RANDOM_TOKEN_ID (5):** The model sees a random word (e.g., "dog") and asks "This looks wrong; tell me the correct word?"
        - **Why?** To force the model to learn that "dog" is *not* the correct word in this context, even though it's a valid word. This prevents the model from simply guessing the most common word in the dictionary.
    - **IGNORE_INDEX (-100):** The model is told to "ignore" these positions.
        - **Why?** Because we only want the model to be penalized for wrong predictions at the masked positions. We don't want to penalize it for the words we intentionally left alone or replaced randomly.
 */
 #define IGNORE_INDEX -100
 #define KEEP_TOKEN_ID -29
 #define RANDOM_TOKEN_ID -30

 /*
    The MLM Head Structure
    ----------------------
    Build the "Head." This is a single matrix of weights ($W_{MLM}$) and a bias vector ($b_{MLM}$).
    ($W_{MLM}$) has shape [d_model. vocab_size] and ($b_{MLM}$) has shape [vocab_size].

    - **Encoder Output:** A [**batch_size**, **sequence_length**, **d_model**] matrix for example [25000, 7, 8], where 25000 is the numbers of lines in the batch, 7 is the number of tokens in each line, and 8 is the dimension of the embedding.
    - **Your Vocabulary Size:** Let's say you have **28** words in your dictionary.
    - **The Weight Matrix:** You need a new matrix (lets call it $W_{MLM}$) of size [**d_model**, **vocab_size**] e.g. [8, 28].
    - **The Bias Vector:** You need a new vector (lets call it $b_{MLM}$) of size [**vocab_size**] e.g. [28].
    - **The Linear Transformation:** You multiply the encoder output by a weight matrix $W_{MLM}$ of size [d_model, vocab_size] (take the [MASK] token as input -> [1, d_model] which is the first token in the sequence, and project it into the size of your Vocabulary -> [1, vocab_size]).
    - **The Softmax:** You turn those raw scores (logits of the [MASK] token) into probabilities (0 to 1) (use a Softmax to get the probability of each word in the Vocabulary).
    - **The Result:** A probability for every word in your dictionary. The word with the highest probability is your "prediction" for the [MASK].

    - **The Cross-Entropy Loss:** PLEASE READ REST OF IT IN THE MLM.md at url https://github.com/sohailqayummalik/BERT/blob/main/Implementation/ML/NLP/transformers/encoder-decoder/DOCUMENTS/MLM.md
 */

/*
    Note: The delimiter used to separate the elements in the COMMAND macro can be customized.
    The first definition uses commas (,) as delimiters, while the second definition uses whitespace. 
    If you wish to change the delimiter or adjust its size, you can modify the corresponding settings in the file...
    lib/csv/parser.h or in your CMakeLists.txt.
    Alternatively, you can undefine and redefine the delimiter after including the lib/argsv-cpp/lib/parser/parser.hh 
    file according to your specific requirements.

    Please note that the macros mentioned below are by default or originally defined in the file lib/csv/parser.h
    #define GRAMMAR_END_OF_TOKEN_MARKER ","
    #define GRAMMAR_END_OF_TOKEN_MARKER_SIZE 1
    #define GRAMMAR_END_OF_LINE_MARKER "\n"
    #define GRAMMAR_END_OF_LINE_MARKER_SIZE 1

    The following two macros are defined in file  lib\argsv-cpp\lib\parser\parser.hh
    #define HELP_STR_START    "("
    #define HELP_STR_END      ")"
 */
/*
    To change the default parsing behaviour of the CSV parser
        
    Snippet from CMakeLists.txt file
    # Add a definition for the GRAMMAR_END_OF_TOKEN_MARKER macro
    #add_definitions(-DGRAMMAR_END_OF_TOKEN_MARKER=" ")
    #add_definitions(-DGRAMMAR_END_OF_TOKEN_MARKER_SIZE=1)

    Snippet from CMakeLists.txt file
    # Add a definition for the GRAMMAR_END_OF_TOKEN_MARKER macro for the replika target
    #target_compile_definitions(replika PRIVATE GRAMMAR_END_OF_TOKEN_MARKER=" ")
    #target_compile_definitions(replika PRIVATE GRAMMAR_END_OF_TOKEN_MARKER_SIZE=1)
 */
/*
    To change the default parsing behaviour of the CSV parser

    Snippet from the msbuild project file(named here project.xml)
    <ItemDefinitionGroup>
        <ClCompile>
            <PreprocessorDefinitions Condition="'$(CSVPreprocessorDefinitions)'=='yes'">CSV_EXAMPLE_APPLICATION;CSV_NOT_ALLOW_EMPTY_TOKENS;GRAMMAR_END_OF_TOKEN_MARKER=" "</PreprocessorDefinitions>
        </ClCompile>
    </ItemDefinitionGroup>  

    and then youe compile...
    @msbuild project.xml /p:CSVPreprocessorDefinitions=yes
 */

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#define GRAMMAR_END_OF_TOKEN_MARKER " "

#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 8

#endif

#include "./../../Implementation/MLM/mlm.hh"

