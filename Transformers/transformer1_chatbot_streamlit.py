# 1) Data Input
# transformer models do not process data sequentially like RNNs.
# instead, they process data in parallel using a technique called self-attention,
# which allows the model to attend to all input tokens at once.

# 2) Embedding Input + Positional Encoding
# The positional encoding is typically added to the input embeddings before
# they are fed into the transformer model. The encoding is based on a set of
# sine and cosine functions that encode the
# position of each token along different dimensions.

# 3) Multi-Head Atteniton
# the input sequence is first transformed into three separate vectors:
# query, key, and value. Each of these vectors is then used in parallel in
# multiple attention heads to compute a set of weighted sums for each of the three vectors.
# The outputs of these attention heads are concatenated and passed
# through a linear layer to produce the final output.

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Initialize the chat history
chat_history_ids = None


# Define the Streamlit app
def main():
    st.title("DialoGPT Chatbot")

    # Create a text input for the user to input messages
    user_input = st.text_input("User Input:")

    if user_input:
        # Encode the user input and add the end-of-sequence token
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, return_tensors="pt"
        )

        # Append the user input tokens to the chat history
        global chat_history_ids
        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            if chat_history_ids is not None
            else new_user_input_ids
        )

        # Generate a response to the chat history
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated response and display it to the user
        bot_response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
        )
        st.text_area("DialoGPT:", value=bot_response)


if __name__ == "__main__":
    main()
