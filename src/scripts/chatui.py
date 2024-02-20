from chatbot import Chatbot
import streamlit as st


st.title("ChatGPT-like clone")

model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

client = Chatbot(
    model=model,
    system_prompt="You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience.",
    streaming=True,
)

# st.write(f"You selected: {model} and the chatbot model now is {client.model}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    client.add_user_prompt(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.generate_response()
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
