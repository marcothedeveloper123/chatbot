from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit import ANSI
from prompt_toolkit.styles import Style
import argparse
from openai import OpenAI


class Chatbot:
    def __init__(
        self,
        system_prompt="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        model="gpt-3.5-turbo",
    ):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model = model
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
        response = self.client.chat.completions.create(
            model=self.model, messages=self.conversation_history
        )
        response_text = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        if __name__ != "__main__":
            return response_text
        else:
            print(f"\n{self.colors['BLUE']}{response_text}{self.colors['RESET']}\n")

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
                else:
                    self.add_user_prompt(user_prompt)
                    self.generate_response()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{self.colors['RESET']}Exiting...")


if __name__ == "__main__":
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
    args = parser.parse_args()

    chatbot = Chatbot(system_prompt=args.system, model=args.model)
    try:
        if args.user:
            user_prompt = chatbot.add_user_prompt(args.user)
            chatbot.generate_response()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{chatbot.colors['RESET']}Exiting...")
    chatbot._run()
