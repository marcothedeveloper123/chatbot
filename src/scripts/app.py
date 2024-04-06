from flask import Flask, render_template
from flask_socketio import SocketIO
from chatbot import Chatbot

app = Flask(__name__)
socketio = SocketIO(app)

chatbot = Chatbot(model="mistral:7b-instruct-v0.2-fp16", streaming=True)

@app.route('/')
def index():
	return render_template('index.html')

@socketio.on('send_message')
def handle_message(data):
	user_message = data['message']
	chatbot.add_prompt_to_conversation("user", user_message)

	# Stream response from the chatbot
	for response in chatbot.stream_response():
		if isinstance(response, dict) and response.get("end_of_stream"):
			# When stream_response yields None, it signals the end of a response
			socketio.emit('end_chatbot_response', {})  # This can be used to trigger frontend to prepare for next message
		else:
			# Regular message content, emit to the client
			socketio.emit('chatbot_response', {'message': response})

if __name__ == '__main__':
	socketio.run(app, debug=True)
