import torch
import torch.nn.functional as func

#lets test a prediction value giving softmax
predictions = torch.log(torch.rand(3))
print(predictions)

# lets obtain the predictions softmax value
print(func.softmax(predictions, dim=-1))
# giving the first index the hiest value of the softmax function
print(torch.argmax(func.softmax(predictions, dim=-1)))

# now we can increase prediction using 
# temperatur by using a divisor of less than 1
# comparing with and WITHOUT temperature
# NOTE: first index gets pushed even higher
# now distribution is much more confidant
print(func.softmax(predictions, dim=-1))
print(func.softmax(predictions / 0.5, dim=-1))

# lets compare iwth lower values
print(func.softmax(predictions, dim=-1))
print(func.softmax(predictions / 0.2, dim=-1))