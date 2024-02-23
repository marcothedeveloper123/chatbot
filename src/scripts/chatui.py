from chatbot import Chatbot
import streamlit as st


def initiate_session_variables(chatbot):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = chatbot.system_prompt


def get_model_options(chatbot):
    formatted_models = []
    for chatbot, models in chatbot.models_cache.items():
        formatted_models.extend([f"{chatbot}: {model}" for model in models])
    return formatted_models


def select_model(chatbot):
    model_options = get_model_options(chatbot)
    selected_option = st.sidebar.selectbox("Model", model_options)
    client, model = selected_option.split(": ", 1)
    st.sidebar.text(model)

    chatbot.model = model


def set_system_prompt(chatbot):
    # Function to directly handle system prompt updates based on text area input
    current_prompt = (
        chatbot.system_prompt
    )  # Get the current system prompt from the chatbot

    # Create a text area for the system prompt, using the current system prompt as the default value
    updated_prompt = st.sidebar.text_area(
        "Enter your new system prompt:", value=current_prompt
    )

    # Check if the system prompt has been updated (i.e., if it differs from the current prompt)
    if updated_prompt != current_prompt:
        chatbot.update_system_prompt(
            updated_prompt
        )  # Update the chatbot's system prompt with the new value

    # Optionally, you can display the current (possibly updated) system prompt for confirmation
    st.sidebar.write(chatbot.system_prompt)


def display_previous_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(chatbot):
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chatbot.add_user_prompt(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = chatbot.stream_response()
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    chatbot = Chatbot(
        system_prompt="You are a helpful assistant",
        # system_prompt="You are a poetic scientist",
        # system_prompt="You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience.",
        streaming=True,
    )
    st.title("ChatGPT-like clone")
    select_model(chatbot)
    initiate_session_variables(chatbot)
    set_system_prompt(chatbot)
    display_previous_messages()
    handle_user_input(chatbot)


if __name__ == "__main__":
    main()
