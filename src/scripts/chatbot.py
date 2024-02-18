import sys
from openai import OpenAI

# ANSI codes to prettify the terminal text
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"

try:
  # check if a command line argument is provided
  if len(sys.argv) < 2:
    user_prompt = input(f"{YELLOW}>>> ")
  else:
    user_prompt = sys.argv[1]

  client = OpenAI()

  conversation_history = [
      {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
      {"role": "user", "content": user_prompt}
    ]

  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation_history
  )

  print(f"\n{BLUE}{response.choices[0].message.content}{RESET}\n")

  conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
except (EOFError, KeyboardInterrupt):
  print(f"\n{RESET}Exiting...")
  sys.exit()

while True:
  try:
    user_prompt = input(f"{YELLOW}>>> ")

    conversation_history.append({"role": "user", "content": user_prompt})

    # Now when the user prompts the AI again, include the conversation history
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=conversation_history
    )

    print(f"\n{BLUE}{response.choices[0].message.content}{RESET}\n")

    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
  except (EOFError, KeyboardInterrupt):
     print(f"\n{RESET}Exiting...")
     break  # exit the loop