import sys
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# check if a command line argument is provided
if len(sys.argv) < 2:
    print('Usage: python chatbot.py "<Your prompt here>"')
    sys.exit(1)

user_prompt = sys.argv[1]

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        },
        {"role": "user", "content": user_prompt},
    ],
)

print(completion.choices[0].message.content)
