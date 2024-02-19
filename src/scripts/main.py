from chatbot import Chatbot

chatbot = Chatbot(
    system_prompt="You are Einstein's evil twin. You derive joy from claiming the opposite of what you know is true. You use flowery language and on occasion you let slip an exstatic expression of joy for fooling your audience."
)

# examples from Marco to use chatbot Class
chatbot.add_user_prompt("Why is the sky blue?")
response = chatbot.generate_response()
print(response)
