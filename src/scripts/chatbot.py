import sys
import argparse
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# ANSI codes to prettify the terminal text
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Chatbot that generates responses based on a system and user prompt."
)
parser.add_argument(
    "-s",
    "--system",
    type=str,
    help="System prompt",
    default="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
)
parser.add_argument("-u", "--user", type=str, help="User prompt", default=None)

# Parse arguments
args = parser.parse_args()

client = OpenAI()
system_prompt = args.system
conversation_history = []

try:
    # Initial user prompt handling
    user_prompt = args.user if args.user is not None else input(f"{YELLOW}>>> ")
    if user_prompt.strip().startswith("-s"):
        system_prompt = user_prompt[3:].strip()  # Update system prompt
        conversation_history.append({"role": "system", "content": system_prompt})
        print(f"\n{MAGENTA}System prompt updated: {system_prompt}{RESET}\n")
    else:
        conversation_history.append({"role": "system", "content": system_prompt})
        conversation_history.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=conversation_history
        )
        print(f"\n{BLUE}{response.choices[0].message.content}{RESET}\n")
        conversation_history.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
except (EOFError, KeyboardInterrupt):
    print(f"\n{RESET}Exiting...")
    sys.exit()

while True:
    try:
        user_prompt = input(f"{YELLOW}>>> ")
        if user_prompt.strip().startswith("-s"):
            system_prompt = user_prompt[3:].strip()  # Update system prompt
            print(f"\n{MAGENTA}System prompt updated: {system_prompt}{RESET}\n")
            # Insert a new system prompt into the conversation history
            # conversation_history.append({"role": "system", "content": system_prompt})
            conversation_history[0] = {"role": "system", "content": system_prompt}
            conversation_history.append({"role": "system", "content": system_prompt})
        else:
            conversation_history.append({"role": "user", "content": user_prompt})
            # Now when the user prompts the AI again, include the conversation history
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=conversation_history
            )

            print(f"\n{BLUE}{response.choices[0].message.content}{RESET}\n")
            conversation_history.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
    except (EOFError, KeyboardInterrupt):
        print(f"\n{RESET}Exiting...")
        break  # exit the loop
