# Elevenlabs ist ausgeschaltet (kein Credit mehr)
# Authentication ist ausgeschaltet (funktioniert leider nicht richtig bei refresh)

# ------------------- Imports --------------------------------------------------------------------------------
# region
# Authentication Keys
from dotenv import find_dotenv, load_dotenv  # environment keys - need to be put in .env

# Password Security
from pathlib import WindowsPath, Path
import streamlit_authenticator as stauth
import yaml  # password data are now in a yaml file
from yaml.loader import SafeLoader

# import pickle
# from passlib.hash import bcrypt

# Steamlit
import streamlit as st

# Openai
from openai import OpenAI

# Jan.AI - Ollama
import requests
import ollama  # not used yet, since using Jan at the moment

# Google
import google.generativeai as genai

# Text to speech
import os
from elevenlabs import generate
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Speech to text
from st_audiorec import st_audiorec

# endregion

# ------------------- Funktionen zum Erstellen der AI-Antwort und dem Zusammenfügen der Ketten ---------------
# region


### ------- Funktion für HTTP LLM API call (Local LLMs through Jan / Ollama) ----------------------------
def llm_http_api_call(messages):

    URL = "http://127.0.0.1:11434/v1/chat/completions"  # 1337 for jan.ai

    payload = {
        # "model": "tinyllama-1.1b", # for jan.ai
        # "model": "orca-mini:latest", # for ollama
        "model": "tinyllama:latest",  # for ollama
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
        "max_tokens": 2048,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    response = requests.post(URL, headers="", json=payload, stream=False)
    reply = response.json()["choices"][0]["message"]["content"]
    return reply


### ------- Funktion für Python OpenAI API call ------------------------------------------------
def llm_function_openai(message, messages, messages_ui, model="OpenGPT"):
    messages.append(
        {"role": "user", "content": message},
    )
    if model == "OpenGPT":
        chat = llm.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, stream=False
        )
        reply = chat.choices[0].message.content
    else:
        reply = llm_http_api_call(messages)

    messages.append({"role": "assistant", "content": reply})
    messages_ui = (
        messages_ui + "\n" + "USER:   " + message + "\n" + "CHATBOT:   " + reply + "\n"
    )

    return messages, messages_ui, reply


### ------- Funktion für Gemini API call -------------------------------------------------------
def llm_function_gemini(message):
    # problem with Gemini: Safety rules - Hate speech
    response = chat_gemini.send_message(message)
    messages_ui = "\n".join(
        [
            (message.role + ":   " + message.parts[0].text)
            for message in chat_gemini.history
        ]
    )
    reply = response.candidates[0].content.parts[0].text
    return chat_gemini.history, messages_ui, reply


def create_output():
    # st.write("wird erreicht")
    reply = ""
    if (
        st.session_state.user_input
        and st.session_state.user_input != st.session_state.old_user_input
    ):
        st.session_state.old_user_input = st.session_state.user_input
        if llm_selection == "OpenGPT" or llm_selection == "Ollama":
            st.session_state.messages, st.session_state.messages_ui, reply = (
                llm_function_openai(
                    st.session_state.user_input,
                    st.session_state.messages,
                    st.session_state.messages_ui,
                    st.session_state.llm_selection,
                )
            )
        elif llm_selection == "Gemini":
            st.session_state.messages, st.session_state.messages_ui, reply = (
                llm_function_gemini(st.session_state.user_input)
            )
    return st.session_state.messages_ui, reply


def create_play_audio(reply, voice):
    # Test auf Voice-Output

    ttsAudio = OpenAI().audio.speech.create(
        model="tts-1", voice=voice, input=reply, response_format="mp3"
    )
    ttsAudio.write_to_file(r"ttsModel.mp3")  # potential error if starting from home
    # autoplay_audio(mp3_path)
    # Streamlit Audio
    if os.path.exists(mp3_path):
        audio_file = open(mp3_path, "rb")
        st.write("OpenAI speech synthesis:")
        st.audio(audio_file.read(), format="audio/mp3")

    st.write("Elevenlabs speech synthesis - taken out because of lack of quota")
    # st.audio(elevenlabs_tts_new("text"), format="audio/mp3")
    # st.audio(elevenlabs_tts_old("text"), format="audio/mp3")

    st.markdown(
        "**transcript:** \n" + create_transscript("ttsModel.mp3")
    )  # create_transscript("ttsModel.mp3")

    return

    # def autoplay_audio(file_path: str):
    # with open(file_path, "rb") as f:
    #     data = f.read()
    #     b64 = base64.b64encode(data).decode()
    #     md = f"""
    #         <audio controls autoplay="true">
    #         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    #         </audio>
    #         """
    #     st.markdown(md, unsafe_allow_html=True)


def create_transscript(soundfile):
    # myobj = gTTS(text=soundfile, lang="en", slow=False)
    # myobj.save("open-ai-library\transcript.mp3")
    audio_file = open(soundfile, "rb")
    transcript = OpenAI().audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="text"
    )
    return transcript


def change_model():
    llm_selected = st.session_state.llm_selection
    voice_selected = st.session_state.voice_selection
    st.session_state.clear()
    st.session_state.llm_selection = llm_selected
    st.session_state.voice_selection = voice_selected
    st.session_state.authentication_status = True


def elevenlabs_tts_old(text):
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_KEY"))
    audio = generate(
        text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
        voice="Rachel",
        model="eleven_multilingual_v2",
    )
    play(audio)
    return audio


### sehr schwierig, dass Ganze, der Output ist ein python-generator objekt, ein stream mit chunks, nicht ein mp3
### der Generator muss zuerst in einer speziellen Datei gespeichert werden, um dann von st.audio gelesen werden zu können
def elevenlabs_tts_new(text):
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_KEY"))
    audio = client.text_to_speech.convert(
        text="Hola?",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with open(r"ttsModel.mp3", "wb") as f:
        for chunk in audio:
            f.write(chunk)
    return r"ttsModel.mp3"


# endregion

# ------------------- Web Page Setup + CSS -------------------------------------------------------------------
# region

### Setting the page
st.set_page_config(
    page_title="Chadbot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # custom menu items, leider kann der Menu-Item-Text nicht geändert werden
    menu_items={
        "Get Help": "https://www.google.de",
        "Report a bug": "https://www.wikipedia.org",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

# -------------------- Setting the backgrounds with CSS -------------------
# from  https://www.magicpattern.design/tools/css-backgrounds

#### constants
gcolor_m = "#eaffea"
gcolor_s = "#11ddb4"
gcolor_h = "#FFFFFF"

#### Setting the background
global_css = f"""
<style>
[data-testid="stApp"] {{
background-color: #ffffff;
opacity: 1;
background-image:  linear-gradient(135deg, {gcolor_m} 25%, transparent 25%), linear-gradient(225deg, {gcolor_m} 25%, transparent 25%), linear-gradient(45deg, {gcolor_m} 25%, transparent 25%), linear-gradient(315deg, {gcolor_m} 25%, #ffffff 25%);
background-position:  4px 0, 4px 0, 0 0, 0 0;
background-size: 4px 4px;
background-repeat: repeat;
}}

[data-testid="stSidebarContent"] {{
background-color: #ffffff;
opacity: 1;
background-image:  linear-gradient(135deg, {gcolor_s} 25%, transparent 25%), linear-gradient(225deg, {gcolor_s} 25%, transparent 25%), linear-gradient(45deg, {gcolor_s} 25%, transparent 25%), linear-gradient(315deg, {gcolor_s} 25%, #ffffff 25%);
background-position:  4px 0, 4px 0, 0 0, 0 0;
background-size: 4px 4px;
background-repeat: repeat;
}}

[data-testid="stHeader"] {{
background-color: #777777;
opacity: 1;
}}
</style>

"""
st.markdown(global_css, unsafe_allow_html=True)

# delete leading gap
# source: https://discuss.streamlit.io/t/leading-gap/41581/2
st.markdown(
    """
<style>
div[class^="block-container"] {
    padding-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# endregion

# ------------------- Globale Variablen, Initialisieren von  Umgebungsvariablen und Öffnen der LLMs ----------
# region
load_dotenv(find_dotenv())
software_comment = "Implementation of a chatbot in Python using the OpenAI Python module (Chat.completions Python). Frontend in Streamlit, utilizing session states. Extended by also including routines to connect to Google Gemini and Ollama Local LLM Models"
# system_prompt = "You are an arrogant prick named Chad. You are looking down to people not as rich and good-looking as you and you think life is easy. And you are making jokes about it"
system_prompt = "You are a helpful agent"
start_message = "Hello, how is life going?"
# output_ui = ""
reply = ""
mp3_path = os.path.join(os.getcwd(), "ttsModel.mp3")


############## Open-AI - Start der Message-Ketten - Definition von Systemprompt und Startmessage, falls erste Iteration
if "old_user_input" not in st.session_state:
    st.session_state.old_user_input = (
        "42 - the answer to life, the universe, and everything - QWERTZ"
    )

if "llm_selection" not in st.session_state:
    st.session_state.llm_selection = "OpenGPT"  ### to make it possible to work with the variable even on first instance (because radio selection not loaded yet)

if "voice_selection" not in st.session_state:
    st.session_state.voice_selection = "alloy"

if "messages_ui" not in st.session_state:
    if (
        st.session_state.llm_selection == "OpenGPT"
        or st.session_state.llm_selection == "Ollama"
    ):
        st.session_state.messages_ui = "ASSISTANT" + ":   " + system_prompt
        st.session_state.messages_ui = (
            st.session_state.messages_ui + "\n" + "CHATBOT:   " + start_message + "\n"
        )
    elif st.session_state.llm_selection == "Gemini":
        st.session_state.messages_ui = "user" + ":   " + system_prompt
        st.session_state.messages_ui = (
            st.session_state.messages_ui + "\n" + "model:   " + start_message + "\n"
        )

if "messages" not in st.session_state:
    if (
        st.session_state.llm_selection == "OpenGPT"
        or st.session_state.llm_selection == "Ollama"
    ):
        st.session_state.messages = [
            {"role": "system", "content": system_prompt},
        ]
        st.session_state.messages.append(
            {"role": "assistant", "content": start_message},
        )
    elif st.session_state.llm_selection == "Gemini":
        gemini_initial_history = [
            {
                "role": "user",
                "parts": [{"text": "System prompt: " + system_prompt}],
            },
            {
                "role": "model",
                "parts": [{"text": start_message}],
            },
        ]
        st.session_state.messages = gemini_initial_history

if st.session_state.llm_selection == "OpenGPT":
    llm = OpenAI()
    model_used_message = "Wow, you are using OpenAI!"

elif st.session_state.llm_selection == "Gemini":
    model = genai.GenerativeModel("gemini-pro")
    chat_gemini = model.start_chat(history=st.session_state.messages)
    model_used_message = "Wow, you are using Gemini!"

else:
    model_used_message = "Wow, you are using Ollama!"

# endregion


# ------------------- User Authentication ---------------------------------------------------------------------
# region

if "first_run" not in st.session_state:
    st.session_state.first_run = True

# muss immer geladen werden, um das Authenticator Object immer da zu haben und mit ihm Aktionen ausführen zu können (z.B. Logout)
# status_authentication status wird auf None gesetzt
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )

# Login Schirm. Eingabefelder werden nur angezeigt, übersprungen, wenn Cookie oder positiver authentication Status
# In dem Abgleich wird die st.session_state.authentication_status gesetzt
name, authentication_status, username = authenticator.login()

# Debugs
# "First_run_flagg:", st.session_state.first_run
# "Authentication_status:", st.session_state.authentication_status


# Meldungen bei Fehlanmeldung
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")


# Weitermachen nur bei erfolgreichem Login
if (
    st.session_state.authentication_status == True
    and st.session_state.first_run == False
):

    # -------------------  Streamlit Interface --------------------------------------------------------------------
    # region

    # -------- Sidebar -----------------------------
    # region

    st.sidebar.title("Parameter-Input")

    llm_selection = st.sidebar.radio(
        label="LLM-Modell",
        options=["OpenGPT", "Gemini", "Ollama"],
        key="llm_selection",
        on_change=change_model,
        horizontal=True,
    )

    voice_selection = st.sidebar.radio(
        label="Voice",
        options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        key="voice_selection",
        horizontal=True,
    )

    user_input = st.sidebar.text_input(
        label="User Input",
        help="Schreiben Sie hier ihre Frage hinein",
        key="user_input",
        # on_submit=create_output,
    )

    col1, col2 = st.sidebar.columns([5, 5], gap="small")

    ### Reset Chatbot
    clear_model = col1.button(
        label="Clear Chat",
        help="Clears the message history",
        on_click=change_model,
        use_container_width=True,
    )

    ### Logout - Manual try - does not delete the cookie
    # clear_model = col1.button(
    #     label="Logout",
    #     help="Logout to login again",
    #     on_click=logout,
    #     use_container_width=True,
    # )

    ### Download Chat History
    download_button = col2.download_button(
        label="Download chat",
        data=st.session_state.messages_ui,
        file_name="chat.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # logout granular angesprochen, wegen Formatierung des Buttons und setzen eines Flaggs
    if col1.button(
        label="Logout", help="Logout to login again", use_container_width=True
    ):
        st.session_state.clear()
        st.session_state.first_run = True
        st.rerun()

    # ------------ stt diabled so far ----------------
    # wav_audio_data = st_audiorec()
    # if wav_audio_data is not None:
    #     st.sidebar.audio(wav_audio_data, format="audio/wav")
    #     # st.sidebar.write(wav_audio_dat

    # sidebar_container = st.sidebar.container(border=True)
    # endregion

    # --------- Main-area ---------------------------
    # region
    st.title(":sunglasses: ChadGPT :sunglasses: ", anchor="ChadGPT")
    # st.write(f"Hello {name}!") # taken out for bug_fixing - correct and put back in
    st.write(software_comment)
    st.write(model_used_message)

    if user_input:
        st.session_state.messages_ui, reply = create_output()
    text_area = st.text_area(
        "Chat: ", value=st.session_state.messages_ui, height=300, key="text_output"
    )

    # enthält die ganzen Audio-Spielereien
    if reply:
        create_play_audio(reply, voice_selection)

    # debugging
    # st.text_area("Session states:", value=st.session_state, height=300)

    # endregion
    # endregion

if authentication_status == True and st.session_state.first_run == True:
    st.session_state.first_run = False
    st.rerun()

    # -------------------  Streamlit Backup / other stuff ---------------------------------------------------------
    # region
    ### Sidebars
    # st.sidebar.title("Navigation")
    # pages = ["ChatBot", "Seite2", "Seite3", "Seite4"]
    # page = st.sidebar.radio("Zu Seite ...", pages)
    # if page == pages[0]:
    # st.title("Introduction")

    ### Widgets
    # st.selectbox(label="Seite", options=["Option1", "Option2"])
    # st.image("energie.jpg", use_column_width=True)
    # order_options = ["Lineaire", "Polynomiale"]
    # order_selection = st.radio("Ordre de la regréssion", order_options)

    # if st.checkbox("Afficher la courbe de consommation mensuel"):
    #     st.write("Hurra!")
    # else:
    #     st.write("Nicht genug!")

    # Caching - did not work as intended
    # @st.cache_data
    # def load_env_setup():
    # global system_prompt
    # global start_message
    # load_dotenv(find_dotenv())
    # system_prompt = "You are an arrogant prick named Chatbot. You are looking down to people not as rich and good-looking as you and you think life is easy. And you are making jokes about it"
    # start_message = "Hello, how is life going?"

    # @st.cache_resource
    # def initialize_openai():
    #     return OpenAI()
    # load_env_setup()
    # llm = initialize_openai()
    # endregion
