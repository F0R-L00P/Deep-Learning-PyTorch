from transformers import pipeline, set_seed

# set seed for reproducibility
set_seed(8)
# setup the task and the model to be used
test_generator = pipeline("text-generation", model="gpt2")

# check model, to complete sentece with length and return sequences
replies = test_generator(
    "Hello, I'm a Language Model, and I think", max_length=30, num_return_sequences=5
)

# loop over values and print generated texts
for dictionary in replies:
    values = dictionary.values()
    print(*values, sep="\n")
