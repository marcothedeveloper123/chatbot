from chatbot import Chatbot

chatbot = Chatbot(
    model="openhermes:latest",
    # model="gpt-3.5-turbo",
    system_prompt="You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience.",
    streaming=True,
)

chatbot.add_user_prompt("Why is the sky blue?")

for response_text in chatbot.generate_response():
    print(
        response_text,
        end="",
        flush=True,
    )

# response = chatbot.generate_response()
# print(response)
