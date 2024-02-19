# code from Streamlit itself - shows how to use ChatElements
import streamlit as st
import os
from openai import OpenAI
from chatbot import Chatbot
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

system_prompt = "You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience."

# chatbot = Chatbot(system_prompt=system_prompt)
# chatbot.add_user_prompt("Why is the sky blue?")
# response = chatbot.generate_response()

with st.sidebar:
    if not openai_api_key:
        openai_api_key = st.text_input(
            "OpenAI API Key", key="chatbot_api_key", type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")

# Initialization of session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "How can I help you?"},
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input -> Add new items to chat history window, update session state
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # _# replace
    # chatbot = Chatbot(system_prompt=system_prompt) #problem: how do I add conversation history?
    client = OpenAI(api_key=openai_api_key)

    # _# replace
    # Take out for version without a conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("user").write(prompt)

    # chatbot.add_user_prompt("Why is the sky blue?")
    # response = chatbot.generate_response()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=st.session_state.messages
    )
    msg = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
