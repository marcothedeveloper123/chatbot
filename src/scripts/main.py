def _run(self):  # we only call this from the command line
try:
    while True:
        style = Style.from_dict(
            {
                "prompt": "ansiyellow",  # Style for the prompt text
                "": "ansiyellow",  # Style for the default (user input) text
            }
        )
        user_prompt = prompt(  # present a command line prompt for the user to write their prompt to the LLM
            ANSI(f"{self.colors['YELLOW']}>>> "),
            history=chatbot.history,
            style=style,
        )
        if user_prompt.startswith(
            "-s"
        ):  # if the user starts the prompt with `-s`, we change the system prompt instead
            self.system_prompt = user_prompt[3:].strip()
            self.conversation_history.append(
                {"role": "system", "content": self.system_prompt}
            )
            if (
                __name__ == "__main__"
            ):  # if we run this script from the command line, confirm the system prompt change
                """
                TO DO: is this `if` clause even necessary if we call `-run()` only from the command line?
                """
                print(
                    f"\n{self.colors['MAGENTA']}System prompt updated: {self.system_prompt}{self.colors['RESET']}\n"
                )
                print(f"{self.colors['RESET']}\n")
        else:
            self.add_user_prompt(user_prompt)
            """
            the user prompt does not start with `-s`, so we treat it as a prompt to the LLM
            """
            if (
                chatbot.streaming
            ):  # we run streaming reponses through a `for` loop
                print(f"{chatbot.colors['BLUE']}")
                for (
                    response_text
                ) in (
                    chatbot.stream_response()
                ):  # here, `generate_response()` returns a generator object
                    print(
                        response_text,
                        end="",  # if we do this, the `print` command will keep printing on the same line
                        flush=True,  # flush the output buffer after each print to ensure immediate display
                    )
                print(f"{chatbot.colors['BLUE']}")
            else:
                response_text = (
                    chatbot.generate_response()
                )  # here, `generate_response` returns a string
                print(f"\n{self.colors['BLUE']}{response_text}")
            print(f"{self.colors['RESET']}\n")
except (
    EOFError,
    KeyboardInterrupt,
):  # exit gracefully when the user presses `Ctrl+C` or `Ctrl+D`
    print(f"\n{self.colors['RESET']}Exiting...")


load_dotenv(find_dotenv())
"""
If you store your OpenAI API key in the `.env` file in the root folder, the line above will retrieve it.
If you store it in `~/.zshrc`, the OpenAI() object will retrieve it automatically
"""

parser = argparse.ArgumentParser(  # provide some cool command line arguments
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

chatbot = (
    Chatbot(  # generate the Chatbot() object that mediates between user and LLM
        system_prompt=args.system, model=args.model, streaming=args.streaming
    )
)
"""
The `model` parameter tells `Chatbot()` if you want an OpenAI model or an Ollama model.
Here's how it works:
1. We retrieve the list of available OpenAI and Ollama models.
2. Then we compare the `model` parameter to whatever we retrieved (first OpenAI, then Ollama).
3. Depending on where we find the model, we establish the connection to that service.
4. If we don't find the model, we return an error message.
"""
try:
    if args.user:
        user_prompt = chatbot.add_user_prompt(
            args.user
        )  # `-u` in the command line passes a user prompt
        if chatbot.streaming:
            print(f"{chatbot.colors['BLUE']}")
            for response_text in chatbot.stream_response():
                print(
                    response_text,
                    end="",
                    flush=True,
                )
            print(f"{chatbot.colors['RESET']}\n")
        else:
            response_text = chatbot.generate_response()
            print(
                f"\n{chatbot.colors['BLUE']}{response_text}{chatbot.colors['RESET']}\n"
            )
except (
    EOFError,
    KeyboardInterrupt,
):  # exit gracefully when the user presses `Ctrl+C` or `Ctrl+D`
    print(f"\n{chatbot.colors['RESET']}Exiting...")
chatbot._run()
