from chatbot import Chatbot
from prompt_toolkit import prompt, ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
import argparse
from dotenv import load_dotenv, find_dotenv
import sys

# Define console colors
colors = {
    "YELLOW": "\033[33m",
    "GREEN": "\033[32m",
    "MAGENTA": "\033[35m",
    "BLUE": "\033[34m",
    "RESET": "\033[0m",
}


def get_user_model_choice(available_models):
    print("Please select a model from the following available models:")
    for idx, model in enumerate(available_models, start=1):
        print(f"{idx}. {model}")
    print(
        f"\nEnter the number of the model you wish to use, or 'exit' to quit:{colors['YELLOW']}"
    )

    while True:
        choice = input(f">>> {colors['RESET']}").strip().lower()
        if choice == "exit":
            sys.exit("Exiting program.")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                return available_models[choice_idx]
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Please enter a valid number or 'exit'.")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{colors['RESET']}Exiting...")


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
    parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="Display token counts for system prompt, user input, and conversation history",
    )
    args = parser.parse_args()

    while True:
        try:
            chatbot = Chatbot(
                system_prompt=args.system, model=args.model, streaming=args.streaming
            )

            # print(f"chatbot.initial_state = {chatbot.initial_state}")

            if chatbot.initial_state == "service_unavailable":
                print(
                    f"{colors['MAGENTA']}The chatbot service is currently unavailable. Please try again later."
                )
                sys.exit(1)
            elif chatbot.initial_state == "model_not_available":
                print(
                    f"{colors['MAGENTA']}The specified model '{args.model}' is not available.\n"
                )
                available_models = [
                    model
                    for models in chatbot.models_cache.values()
                    for model in models
                ]
                args.model = get_user_model_choice(available_models)
                print(f"{colors['RESET']}\n")
                continue
            break
        except (EOFError, KeyboardInterrupt):
            print(f"\n{colors['RESET']}Exiting...")
            sys.exit()

    # print(f"model: {chatbot.model}")

    cmd_history = InMemoryHistory()

    if args.user:
        chatbot.add_user_prompt(args.user)
        display_response(chatbot)
        return

    try:
        print(f"{colors['RESET']}")
        if args.count:
            system_prompt_token_count = chatbot.client.estimate_token_count(
                chatbot.model, chatbot.system_prompt
            )
            print(
                f"{colors['GREEN']}System prompt: {system_prompt_token_count} tokens{colors['RESET']}\n"
            )

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
                system_prompt_token_count = chatbot.client.estimate_token_count(
                    chatbot.model, chatbot.system_prompt
                )
                print(
                    f"{colors['GREEN']}System prompt: {system_prompt_token_count} tokens{colors['RESET']}\n"
                )
            else:
                chatbot.add_user_prompt(user_input)
                display_response(chatbot, user_input, args.count)
    except (EOFError, KeyboardInterrupt):
        print(f"\n{colors['RESET']}Exiting...")
        # TODO consider adding a stopword, e.g. `--exit`


def display_response(chatbot, user_input=None, show_counts=False):
    # Estimate and print the user prompt token count
    if user_input is not None and show_counts:
        user_prompt_token_count = chatbot.client.estimate_token_count(
            chatbot.model, user_input
        )
        print(
            f"\n{colors['GREEN']}User prompt: {user_prompt_token_count} tokens{colors['RESET']}"
        )

    print(f"{colors['BLUE']}")
    if chatbot.streaming:
        for response_text in chatbot.stream_response():
            print(response_text, end="", flush=True)
    else:
        response_text = chatbot.generate_response()
        print(response_text)
    print(f"{colors['RESET']}")

    # Update and print the conversation history token count after the response
    if show_counts:
        conversation_history_token_count = chatbot.conversation_history_token_count
        print(
            f"{colors['GREEN']}Conversation history: {conversation_history_token_count} tokens{colors['RESET']}\n"
        )


if __name__ == "__main__":
    run_chatbot()
