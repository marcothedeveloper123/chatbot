from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit import ANSI
from prompt_toolkit.styles import Style
import argparse
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


class Chatbot:
    def __init__(
        self,
        system_prompt="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        model="gpt-3.5-turbo",
        streaming=False,
    ):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model = model
        self.streaming = streaming
        self.conversation_history = []
        self.colors = {
            "YELLOW": "\033[33m" if __name__ == "__main__" else "",
            "GREEN": "\033[32m" if __name__ == "__main__" else "",
            "MAGENTA": "\033[35m" if __name__ == "__main__" else "",
            "BLUE": "\033[34m" if __name__ == "__main__" else "",
            "RESET": "\033[0m" if __name__ == "__main__" else "",
        }
        self.history = InMemoryHistory()
        self.init_conversation()

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
        self.system_prompt = new_prompt
        self.conversation_history.append(
            {"role": "system", "content": self.system_prompt}
        )
        if __name__ == "__main__":
            print(
                f"{self.colors['MAGENTA']}System prompt updated: {self.system_prompt}{self.colors['RESET']}"
            )

    def generate_response(self):
        if self.streaming:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response_text = chunk.choices[0].delta.content
                    self.conversation_history.append(
                        {"role": "assistant", "content": response_text}
                    )
                    yield response_text
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=self.conversation_history
            )
            response_text = response.choices[0].message.content
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
            return response_text

    def _run(self):
        try:
            while True:
                style = Style.from_dict(
                    {
                        "prompt": "ansiyellow",  # Style for the prompt text
                        "": "ansiyellow",  # Style for the default (user input) text
                    }
                )
                user_prompt = prompt(
                    ANSI(f"{self.colors['YELLOW']}>>> "),
                    history=chatbot.history,
                    style=style,
                )
                if user_prompt.startswith("-s"):
                    self.system_prompt = user_prompt[3:].strip()
                    self.conversation_history.append(
                        {"role": "system", "content": self.system_prompt}
                    )
                    if __name__ == "__main__":
                        print(
                            f"\n{self.colors['MAGENTA']}System prompt updated: {self.system_prompt}{self.colors['RESET']}\n"
                        )
                        print(f"{self.colors['RESET']}\n")
                else:
                    self.add_user_prompt(user_prompt)
                    if chatbot.streaming:
                        print(f"{chatbot.colors['BLUE']}")
                        for response_text in chatbot.generate_response():
                            print(
                                response_text,
                                end="",
                                flush=True,
                            )
                        print(f"{chatbot.colors['BLUE']}\n")
                    else:
                        response_text = chatbot.generate_response()
                        print(response_text)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{self.colors['RESET']}Exiting...")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Chatbot that generates responses based on a system and user prompt."
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        help="System prompt",
    )
    parser.add_argument("-u", "--user", type=str, default=None, help="User prompt")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt-3.5-turbo", help="LLM model to use"
    )
    parser.add_argument(
        "-t", "--streaming", action="store_true", help="Enable streaming mode"
    )
    args = parser.parse_args()

    chatbot = Chatbot(
        system_prompt=args.system, model=args.model, streaming=args.streaming
    )
    try:
        if args.user:
            user_prompt = chatbot.add_user_prompt(args.user)
            if chatbot.streaming:
                print(f"{chatbot.colors['BLUE']}")
                for response_text in chatbot.generate_response():
                    print(
                        response_text,
                        end="",
                        flush=True,
                    )
                print(f"{chatbot.colors['RESET']}\n")
            else:
                response_text = chatbot.generate_response()
                print(response_text)
    except (EOFError, KeyboardInterrupt):
        print(f"\n{chatbot.colors['RESET']}Exiting...")
    chatbot._run()
