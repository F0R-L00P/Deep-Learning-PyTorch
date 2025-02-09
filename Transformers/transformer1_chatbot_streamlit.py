import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_bot():
    """
    Loads the DialoGPT tokenizer and model from Hugging Face.
    Returns (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    return tokenizer, model


def main():
    # Initialize Streamlit app
    st.title("DialoGPT Chatbot")

    # Load or initialize model + tokenizer here
    tokenizer, model = load_bot()

    # Initialize chat history in session state
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None

    # Create user input box
    user_input = st.chat_input("Type your message here...")

    # If user_input is empty or None, do nothing
    if user_input is None or not user_input.strip():
        return

    # Check for exit commands
    if user_input.lower() in ["bye", "exit", "quit"]:
        st.success("Chatbot session ended.")
        st.session_state.chat_history_ids = None
        st.stop()

    # Encode user input
    new_user_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )

    # Append new input to chat history
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat(
            [st.session_state.chat_history_ids, new_user_input_ids], dim=-1
        )
    else:
        bot_input_ids = new_user_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=min(bot_input_ids.shape[-1] + 50, 1000),
        pad_token_id=tokenizer.eos_token_id,
    )

    # Update session state with new chat history
    st.session_state.chat_history_ids = chat_history_ids

    # Limit chat history size
    MAX_CHAT_HISTORY = 500
    if st.session_state.chat_history_ids.shape[-1] > MAX_CHAT_HISTORY:
        st.session_state.chat_history_ids = st.session_state.chat_history_ids[
            :, -MAX_CHAT_HISTORY:
        ]

    # Decode the last response tokens
    bot_response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    st.markdown(f"**DialoGPT:** {bot_response}")


# run program via streamlit run
if __name__ == "__main__":
    main()
