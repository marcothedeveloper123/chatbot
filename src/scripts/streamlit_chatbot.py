# code from Streamlit itself - shows how to use ChatElements
import streamlit as st
import os
from chatbot import Chatbot
from dotenv import load_dotenv, find_dotenv

# --- Initialization
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

system_prompt = "You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience."
system_prompt = "Be nice to me"

# Initialization of session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
    ]

# --- Sidebar and Output
with st.sidebar:
    if not openai_api_key:
        openai_api_key = st.text_input(
            "OpenAI API Key", key="chatbot_api_key", type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# --- User Input and Response functionality

# User input -> Add new items to chat history window, update session state
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Print input
    st.chat_message("user").write(prompt)

    # Processing - Load chatbot and update history
    chatbot = Chatbot(system_prompt=system_prompt)
    chatbot.conversation_history = st.session_state.messages
    chatbot.add_user_prompt(prompt)
    msg = chatbot.generate_response()
    st.session_state.messages = chatbot.conversation_history

    # Show output
    st.chat_message("assistant").write(msg)
