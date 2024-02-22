from chatbot import Chatbot
import streamlit as st


def model_options(chatbot):
    formatted_models = []
    for chatbot, models in chatbot.models_cache.items():
        formatted_models.extend([f"{chatbot}: {model}" for model in models])
    return formatted_models


st.title("ChatGPT-like clone")

chatbot = Chatbot(
    system_prompt="You are a helpful assistant",
    # system_prompt="You are a poetic scientist",
    # system_prompt="You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience.",
    streaming=True,
)

model_options = model_options(chatbot)
selected_option = st.sidebar.selectbox("Model", model_options)
client, model = selected_option.split(": ", 1)
st.sidebar.text(model)

chatbot.model = model

if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    chatbot.add_user_prompt(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = chatbot.stream_response()
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
