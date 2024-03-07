from chatbot import Chatbot
import streamlit as st


def get_chatbot():
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot(
            system_prompt="You are a helpful assistant.",
            model="gpt-3.5-turbo",
            streaming=True,
        )
    return st.session_state.chatbot


def initiate_session_variables(chatbot):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = chatbot.system_prompt
    if "show_token_counts" not in st.session_state:
        st.session_state["show_token_counts"] = False


def get_model_options(chatbot):
    formatted_models = []
    for chatbot, models in chatbot.models_cache.items():
        formatted_models.extend([f"{chatbot}: {model}" for model in models])
    return formatted_models


def select_model(chatbot):
    model_options = get_model_options(chatbot)
    if model_options:
        selected_option = st.sidebar.selectbox("Model", model_options)
        if selected_option:
            client, model = selected_option.split(": ", 1)
            st.sidebar.text(model)
            chatbot.model = model
        else:
            st.sidebar.warning("Please select a model.")
    else:
        st.sidebar.error("No models are available for selection.")


def set_system_prompt(chatbot):
    prompt_disabled = chatbot.initial_state == "service_unavailable"

    if prompt_disabled:
        st.sidebar.warning(
            "The chatbot service is currently unavailable. System prompt modification is disabled."
        )
        return

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
    input_disabled = chatbot.initial_state == "service_unavailable"

    if input_disabled:
        st.warning(
            "The chatbot service is currently unavailable. Please try again later."
        )
        return

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chatbot.add_user_prompt(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.show_token_counts:
            user_prompt_token_count = chatbot.estimate_token_count(prompt)
            st.caption(f"Token count: {user_prompt_token_count}")

        with st.chat_message("assistant"):
            stream = chatbot.stream_response()
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.show_token_counts:
            conversation_history_count = chatbot.conversation_history_token_count
            st.sidebar.write(
                f"Conversation history tokens: {conversation_history_count}"
            )


def main():
    # chatbot = Chatbot(
    #     system_prompt="You are a helpful assistant",
    #     # system_prompt="You are a poetic scientist",
    #     # system_prompt="You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience.",
    #     streaming=True,
    # )
    chatbot = get_chatbot()
    st.title("ChatGPT-like clone")
    select_model(chatbot)
    initiate_session_variables(chatbot)
    set_system_prompt(chatbot)
    st.sidebar.checkbox("Show Token Counts", value=False, key="show_token_counts")
    display_previous_messages()
    handle_user_input(chatbot)


if __name__ == "__main__":
    main()
