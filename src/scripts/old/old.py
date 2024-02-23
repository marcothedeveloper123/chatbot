from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit import ANSI
from prompt_toolkit.styles import Style
import argparse
from openai import OpenAI
import ollama
from dotenv import load_dotenv, find_dotenv


class Chatbot:
    """
    The chatbot class connects to one of these LLM providers.
    - OpenAI (remote)
    - Ollama (local)

    depending on the name of the model. The default model is "gpt-3.5-turbo" on OpenAI.
    """

    def __init__(
        self,
        system_prompt="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        model="gpt-3.5-turbo",
        streaming=False,
    ):
        self.client = None  # necessary for connecting to an OpenAI LLM
        self.system_prompt = system_prompt
        self.model = model
        self.streaming = streaming
        self.conversation_history = []  # necessary to implement chat memory
        self.colors = {
            "YELLOW": "\033[33m" if __name__ == "__main__" else "",
            "GREEN": "\033[32m" if __name__ == "__main__" else "",
            "MAGENTA": "\033[35m" if __name__ == "__main__" else "",
            "BLUE": "\033[34m" if __name__ == "__main__" else "",
            "RESET": "\033[0m" if __name__ == "__main__" else "",
        }
        self.history = InMemoryHistory()  # necessary for command line memory
        self.models_cache = None  # helper variable to enable model selection
        self.init_client_and_model()  # select an available model
        self.init_conversation()

    def fetch_openai_models(
        self,
    ):  # one of the sources for models that populate the model cache
        client = OpenAI()
        list = client.models.list()
        models = [model.id for model in list]
        return models

    def fetch_ollama_models(
        self,
    ):  # the second source for models that populate the model cache
        list = ollama.list()
        models = [model["name"] for model in list["models"]]
        return models

    def init_client_and_model(self):
        """
        Check if `self.model` is available on OpenAI or Ollama. Based on this,
        we can set up the chatbot according to the protocol of the model provider.
        If the model name does not appear in either list, return an error message.
        """
        if not self.models_cache:
            openai_models = self.fetch_openai_models()
            ollama_models = self.fetch_ollama_models()
            self.models_cache = {"openai": openai_models, "ollama": ollama_models}

        if self.model in self.models_cache["openai"]:
            self.client = OpenAI()
        elif self.model in self.models_cache["ollama"]:
            self.client = "Ollama"
        else:
            raise ValueError(
                "The specified model is not available in OpenAI or Ollama models."
            )

    def init_conversation(self, user_prompt=None):
        self.conversation_history.append(
            {"role": "system", "content": self.system_prompt}
        )
        if user_prompt:
            self.conversation_history.append({"role": "user", "content": user_prompt})

    def add_user_prompt(self, user_prompt):
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.history.append_string(user_prompt)

    def update_system_prompt(self, new_prompt):
        """
        Enables the user to change the system prompt. We do this by replacing the system prompt
        at the very start of the conversation. Not sure if this is the best way.
        """
        self.system_prompt = new_prompt
        self.conversation_history.append(
            {"role": "system", "content": self.system_prompt}
        )
        if __name__ == "__main__":
            print(
                f"{self.colors['MAGENTA']}System prompt updated: {self.system_prompt}{self.colors['RESET']}"
            )

    def generate_response(self):
        # Non-streaming mode: return the complete response as a string
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                model=self.model, messages=self.conversation_history
            )
            response_text = response.choices[0].message.content
        elif self.client == "Ollama":
            response = ollama.chat(model=self.model, messages=self.conversation_history)
            response_text = response["messages"]["content"]

        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response_text,
            }
        )
        return response_text

    def stream_response(self):
        print(f"streaming from {self.client}")
        # Streaming mode: yield each response part as it arrives
        if isinstance(self.client, OpenAI):
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                stream=True,
            )
        elif self.client == "Ollama":
            stream = ollama.chat(
                model=self.model, messages=self.conversation_history, stream=True
            )

        for chunk in stream:
            if isinstance(self.client, OpenAI):
                if chunk.choices[0].delta.content is not None:
                    response_text = chunk.choices[0].delta.content
            elif self.client == "Ollama":
                if chunk["message"]["content"] is not None:
                    response_text = chunk["message"]["content"]
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )
            yield response_text


#     def _run(self):  # we only call this from the command line
#         try:
#             while True:
#                 style = Style.from_dict(
#                     {
#                         "prompt": "ansiyellow",  # Style for the prompt text
#                         "": "ansiyellow",  # Style for the default (user input) text
#                     }
#                 )
#                 user_prompt = prompt(  # present a command line prompt for the user to write their prompt to the LLM
#                     ANSI(f"{self.colors['YELLOW']}>>> "),
#                     history=chatbot.history,
#                     style=style,
#                 )
#                 if user_prompt.startswith(
#                     "-s"
#                 ):  # if the user starts the prompt with `-s`, we change the system prompt instead
#                     self.system_prompt = user_prompt[3:].strip()
#                     self.conversation_history.append(
#                         {"role": "system", "content": self.system_prompt}
#                     )
#                     if (
#                         __name__ == "__main__"
#                     ):  # if we run this script from the command line, confirm the system prompt change
#                         """
#                         TO DO: is this `if` clause even necessary if we call `-run()` only from the command line?
#                         """
#                         print(
#                             f"\n{self.colors['MAGENTA']}System prompt updated: {self.system_prompt}{self.colors['RESET']}\n"
#                         )
#                         print(f"{self.colors['RESET']}\n")
#                 else:
#                     self.add_user_prompt(user_prompt)
#                     """
#                     the user prompt does not start with `-s`, so we treat it as a prompt to the LLM
#                     """
#                     if (
#                         chatbot.streaming
#                     ):  # we run streaming reponses through a `for` loop
#                         print(f"{chatbot.colors['BLUE']}")
#                         for (
#                             response_text
#                         ) in (
#                             chatbot.stream_response()
#                         ):  # here, `generate_response()` returns a generator object
#                             print(
#                                 response_text,
#                                 end="",  # if we do this, the `print` command will keep printing on the same line
#                                 flush=True,  # flush the output buffer after each print to ensure immediate display
#                             )
#                         print(f"{chatbot.colors['BLUE']}")
#                     else:
#                         response_text = (
#                             chatbot.generate_response()
#                         )  # here, `generate_response` returns a string
#                         print(f"\n{self.colors['BLUE']}{response_text}")
#                     print(f"{self.colors['RESET']}\n")
#         except (
#             EOFError,
#             KeyboardInterrupt,
#         ):  # exit gracefully when the user presses `Ctrl+C` or `Ctrl+D`
#             print(f"\n{self.colors['RESET']}Exiting...")


# if (
#     __name__ == "__main__"
# ):  # this part of the code only runs if called from the command line
#     load_dotenv(find_dotenv())
#     """
#     If you store your OpenAI API key in the `.env` file in the root folder, the line above will retrieve it.
#     If you store it in `~/.zshrc`, the OpenAI() object will retrieve it automatically
#     """

#     parser = argparse.ArgumentParser(  # provide some cool command line arguments
#         description="Chatbot that generates responses based on a system and user prompt."
#     )
#     parser.add_argument(
#         "-s",
#         "--system",
#         type=str,
#         default="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
#         help="System prompt",
#     )
#     parser.add_argument("-u", "--user", type=str, default=None, help="User prompt")
#     parser.add_argument(
#         "-m", "--model", type=str, default="gpt-3.5-turbo", help="LLM model to use"
#     )
#     parser.add_argument(
#         "-t", "--streaming", action="store_true", help="Enable streaming mode"
#     )
#     args = parser.parse_args()

#     chatbot = (
#         Chatbot(  # generate the Chatbot() object that mediates between user and LLM
#             system_prompt=args.system, model=args.model, streaming=args.streaming
#         )
#     )
#     """
#     The `model` parameter tells `Chatbot()` if you want an OpenAI model or an Ollama model.
#     Here's how it works:
#     1. We retrieve the list of available OpenAI and Ollama models.
#     2. Then we compare the `model` parameter to whatever we retrieved (first OpenAI, then Ollama).
#     3. Depending on where we find the model, we establish the connection to that service.
#     4. If we don't find the model, we return an error message.
#     """
#     try:
#         if args.user:
#             user_prompt = chatbot.add_user_prompt(
#                 args.user
#             )  # `-u` in the command line passes a user prompt
#             if chatbot.streaming:
#                 print(f"{chatbot.colors['BLUE']}")
#                 for response_text in chatbot.stream_response():
#                     print(
#                         response_text,
#                         end="",
#                         flush=True,
#                     )
#                 print(f"{chatbot.colors['RESET']}\n")
#             else:
#                 response_text = chatbot.generate_response()
#                 print(
#                     f"\n{chatbot.colors['BLUE']}{response_text}{chatbot.colors['RESET']}\n"
#                 )
#     except (
#         EOFError,
#         KeyboardInterrupt,
#     ):  # exit gracefully when the user presses `Ctrl+C` or `Ctrl+D`
#         print(f"\n{chatbot.colors['RESET']}Exiting...")
#     chatbot._run()
