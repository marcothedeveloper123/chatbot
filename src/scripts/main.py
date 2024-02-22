from chatbot import Chatbot
from prompt_toolkit import prompt, ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
import argparse
from dotenv import load_dotenv, find_dotenv

# Define console colors
colors = {
    "YELLOW": "\033[33m",
    "GREEN": "\033[32m",
    "MAGENTA": "\033[35m",
    "BLUE": "\033[34m",
    "RESET": "\033[0m",
}


def run_chatbot():
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

    cmd_history = InMemoryHistory()

    if args.user:
        chatbot.add_user_prompt(args.user)
        display_response(chatbot)
        # print(f"{colors['BLUE']}")
        # if chatbot.streaming:
        #     for response_text in chatbot.stream_response():
        #         print(response_text, end="", flush=True)
        # else:
        #     response_text = chatbot.generate_response()
        #     print(f"{response_text}")
        # print(f"{colors['RESET']}\n")
        return

    try:
        while True:
            style = Style.from_dict({"prompt": "ansiyellow"})
            user_input = prompt(
                ANSI(f"{colors['YELLOW']}>>> "), history=cmd_history, style=style
            )
            if user_input.startswith("-s"):
                chatbot.update_system_prompt(user_input[3:].strip())
                print(
                    f"\n{colors['MAGENTA']}System prompt updated: {chatbot.system_prompt}{colors['RESET']}\n"
                )
            else:
                chatbot.add_user_prompt(user_input)
                display_response(chatbot)
                # if chatbot.streaming:
                #     for response_text in chatbot.stream_response():
                #         print(
                #             f"{colors['BLUE']}{response_text}{colors['RESET']}",
                #             end="",
                #             flush=True,
                #         )
                # else:
                #     response_text = chatbot.generate_response()
                #     print(f"\n{colors['BLUE']}{response_text}{colors['RESET']}\n")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{colors['RESET']}Exiting...")
        # TODO consider adding a stopword, e.g. `--exit`


def display_response(chatbot):
    print(f"{colors['BLUE']}")
    if chatbot.streaming:
        for response_text in chatbot.stream_response():
            print(response_text, end="", flush=True)
    else:
        response_text = chatbot.generate_response()
        print(response_text)
    print(f"{colors['RESET']}\n")


if __name__ == "__main__":
    run_chatbot()
