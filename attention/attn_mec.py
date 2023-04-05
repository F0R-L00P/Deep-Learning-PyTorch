import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def attention(query, keys, values):
    # Calculate the dot product between the query and the keys
    scores = np.dot(query, keys.T)
    
    # Normalize the scores using the softmax function
    attention_weights = softmax(scores)
    
    # Compute the weighted sum of the values
    context = np.dot(attention_weights, values)
    
    return context, attention_weights

# Example inputs
query = np.array([1, 0, 0])
keys = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Apply the attention mechanism
context, attention_weights = attention(query, keys, values)

print("Context:", context)
print("Attention Weights:", attention_weights)
