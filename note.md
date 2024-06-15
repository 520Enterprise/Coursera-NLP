一些函数操作

np.concatenate

np.hstack

np.vstack





The number of parameters in an RNN is the same regardless of the input's length. 



```python
# GRADED FUNCTION: log_perplexity
def log_perplexity(preds, target):
    """
    Function to calculate the log perplexity of a model.

    Args:
        preds (tf.Tensor): Predictions of a list of batches of tensors corresponding to lines of text.
        target (tf.Tensor): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: The log perplexity of the model.
    """
    PADDING_ID = 1
    ### START CODE HERE ###
    
    # Calculate log probabilities for predictions using one-hot encoding
    log_p = np.sum(preds * tf.one_hot(target, depth=preds.shape[-1]), axis= -1) # HINT: tf.one_hot() should replace one of the Nones
    # Identify non-padding elements in the target
    non_pad = 1.0 - np.equal(target, PADDING_ID)          # You should check if the target equals to PADDING_ID
    # Apply non-padding mask to log probabilities to exclude padding
    log_p = log_p * non_pad                             # Get rid of the padding
#     print("log_p:", np.sum(log_p, axis=-1))
#     print("non_pad:", np.sum(non_pad, axis=-1))
    # Calculate the log perplexity by taking the sum of log probabilities and dividing by the sum of non-padding elements
    log_ppx = np.sum(log_p, axis=-1) / np.sum(non_pad, axis=-1) # Remember to set the axis properly when summing up
    # Compute the mean of log perplexity
    log_ppx = np.mean(log_ppx) # Compute the mean of the previous expression
        
    ### END CODE HERE ###
    return -log_ppx
```

注意 `axis=-1`

## LSTM

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/9Ipus-N9SEOBQ0EPNElN-Q_9d5fab4afb4a4a7bbc94b4a7b38426e1_286911293-bfea5256-0bba-49b7-8b50-ab84b45406c3.png?expiry=1718236800000&hmac=V7kWe63lAXFg4SUCbwbEVqYlXJhcA7t75cDg_sFKK34)

**LSTM equations (optional):** For better understanding, take a look at the LSTM equations and relate them to the figure above.

The forget gate: $𝑓=𝜎(𝑊_𝑓[ℎ_𝑡−1;𝑥_𝑡]+𝑏_𝑓)$ (marked with a blue 1)

The input gate: $𝑖=𝜎(𝑊_i[ℎ_𝑡−1;𝑥_𝑡]+𝑏_i)$ (marked with a blue 2)

The gate gate (candidate memory cell): $𝑔=\tanh⁡(𝑊_𝑔[ℎ_𝑡−1;𝑥_𝑡]+𝑏_𝑔)$

The cell state:  $𝑐_𝑡=𝑓⊙𝑐_{𝑡−1}+𝑖⊙𝑔$

The output gate: $𝑜=𝜎(𝑊_𝑜[ℎ_{𝑡−1};𝑥_𝑡]+𝑏_𝑜)$ (marked with a blue 3)

The output of LSTM unit:  $ℎ_𝑡=𝑜_𝑡⊙ \tanh (𝑐_𝑡)$

## NER

```python
# GRADED FUNCTION: NER
def NER(len_tags, vocab_size, embedding_dim = 50):
    """
    Create a Named Entity Recognition (NER) model.

    Parameters:
    len_tags (int): Number of NER tags (output classes).
    vocab_size (int): Vocabulary size.
    embedding_dim (int, optional): Dimension of embedding and LSTM layers (default is 50).

    Returns:
    model (Sequential): NER model.
    """

    ### START CODE HERE ### 

    model = tf.keras.Sequential(name = 'sequential') 
    # Add the tf.keras.layers.Embedding layer. Do not forget to mask out the zeros!
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, mask_zero=True))
    # Add the LSTM layer. Make sure you are passing the right dimension (defined in the docstring above) 
    # and returning every output for the tf.keras.layers.LSTM layer and not the very last one.
    model.add(tf.keras.layers.LSTM(embedding_dim, return_sequences=True))
    # Add the final tf.keras.layers.Dense with the appropriate activation function. Remember you must pass the activation function itself ant not its call!
    # You must use tf.nn.log_softmax instead of tf.nn.log_softmax().
    model.add(tf.keras.layers.Dense(len_tags, activation=tf.nn.log_softmax))
    
    ### END CODE HERE ### 

    return model
```

[tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding): Initializes the embedding layer. An embedding layer in tensorflow will input only **positive integers**.

- `Embedding(input_dim, output_dim, mask_zero = False)`.

- `mask_zero` : 如果 `mask_zero = True` ，则： 1. 应保留值 0 作为掩码值，因为它在训练中会被忽略。2. 您需要在 中 `input_dim` 添加 1，因为现在 Tensorflow 会考虑每个句子中可能会出现一个额外的 0 值。

## Siamese

注意这个损失函数

```python
# GRADED FUNCTION: TripletLossFn
def TripletLossFn(v1, v2,  margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        triplet_loss (numpy.ndarray or Tensor)
    """
   
    ### START CODE HERE ###

    # use `tf.linalg.matmul` to take the dot product of the two batches. 
    # Don't forget to transpose the second argument using `transpose_b=True`
    scores = tf.linalg.matmul(v2, v1, transpose_b=True)
    # calculate new batch size and cast it as the same datatype as scores. 

    batch_size = tf.cast(tf.shape(v1)[0], scores.dtype) 
    # use `tf.linalg.diag_part` to grab the cosine similarity of all positive examples
    positive = tf.linalg.diag_part(scores)
    # subtract the diagonal from scores. You can do this by creating a diagonal matrix with the values 
    # of all positive examples using `tf.linalg.diag`
    negative_zero_on_duplicate = scores - tf.linalg.diag(positive)
    # use `tf.math.reduce_sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)`
    mean_negative = tf.math.reduce_sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    # create a composition of two masks: 
    # the first mask to extract the diagonal elements (make sure you use the variable batch_size here), 
    # the second mask to extract elements in the negative_zero_on_duplicate matrix that are larger than the elements in the diagonal 
    mask_exclude_positives = tf.cast((tf.eye(batch_size)==1)|(negative_zero_on_duplicate > tf.expand_dims(positive,1)),
                                    scores.dtype)
    # multiply `mask_exclude_positives` with 2.0 and subtract it out of `negative_zero_on_duplicate`
    negative_without_positive = negative_zero_on_duplicate - (mask_exclude_positives * 2.0)
    # take the row by row `max` of `negative_without_positive`. 
    # Hint: `tf.math.reduce_max(negative_without_positive, axis = None)`
    closest_negative = tf.math.reduce_max(negative_without_positive, axis=1)
    # compute `tf.maximum` among 0.0 and `A`
    # A = subtract `positive` from `margin` and add `closest_negative` 
    triplet_loss1 = tf.maximum(0.0, margin + closest_negative - positive)
    # compute `tf.maximum` among 0.0 and `B`
    # B = subtract `positive` from `margin` and add `mean_negative` 
    triplet_loss2 = tf.maximum(0.0, margin + mean_negative - positive)
    # add the two losses together and take the `tf.math.reduce_sum` of it
    triplet_loss = tf.math.reduce_sum(triplet_loss1 + triplet_loss2)
    
    ### END CODE HERE ###

    return triplet_loss
```

## Attention

```python
def alignment(encoder_states, decoder_state):
    # First, concatenate the encoder states and the decoder state.
    inputs = np.concatenate((encoder_states, decoder_state.repeat(input_length, axis=0)), axis=1)
    assert inputs.shape == (input_length, 2*hidden_size)
    
    # Matrix multiplication of the concatenated inputs and the first layer, with tanh activation
    activations = np.tanh(np.matmul(inputs, layer_1))
    assert activations.shape == (input_length, attention_size)
    
    # Matrix multiplication of the activations with the second layer. Remember that you don't need tanh here
    scores = np.matmul(activations, layer_2)
    assert scores.shape == (input_length, 1)
    
    return scores

# Run this to test your alignment function
scores = alignment(encoder_states, decoder_state)
print(scores)

def attention(encoder_states, decoder_state):
    """ Example function that calculates attention, returns the context vector 
    
        Arguments:
        encoder_vectors: NxM numpy array, where N is the number of vectors and M is the vector length
        decoder_vector: 1xM numpy array, M is the vector length, much be the same M as encoder_vectors
    """ 
    
    # First, calculate the dot product of each encoder vector with the decoder vector
    scores = alignment(encoder_states, decoder_state)
    
    # Then take the softmax of those scores to get a weight distribution
    weights = softmax(scores)
    
    # Multiply each encoder state by its respective weight
    weighted_scores = encoder_states * weights
    
    # Sum up the weights encoder states
    context = np.sum(weighted_scores, axis=0)
    
    return context

context_vector = attention(encoder_states, decoder_state)
print(context_vector)
```

