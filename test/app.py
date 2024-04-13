from config import MODEL, STREAMING, BASE_URL
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from chatbot import Chatbot

app = Flask(__name__)
socketio = SocketIO(app)

chatbot = Chatbot(model=MODEL, streaming=STREAMING, base_url=BASE_URL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models")
def models():
    available_models = [
        model for models in chatbot.model_cache.values() for model in models
    ]
    return jsonify(
        {"available_models": available_models, "current_model": chatbot.model}
    )


@app.route("/conversation_history", methods=["GET"])
def conversation_history():
    # Retrieve the current conversation history from the Chatbot instance
    history = chatbot.conversation_history
    return jsonify({"conversation_history": history})


@socketio.on("switch_model")
def handle_switch_model(data):
    new_model = data["model"]
    chatbot.model = new_model
    socketio.emit("model_switched", {"model": new_model})


@socketio.on("send_message")
def handle_message(data):
    user_message = data["message"]
    chatbot.add_prompt_to_conversation("user", user_message)

    # Stream response from the chatbot
    for response in chatbot.stream_response():
        if isinstance(response, dict) and response.get("end_of_stream"):
            # When stream_response yields None, it signals the end of a response
            socketio.emit(
                "end_chatbot_response", {}
            )  # This can be used to trigger frontend to prepare for next message
        else:
            # Regular message content, emit to the client
            socketio.emit("chatbot_response", {"message": response})


if __name__ == "__main__":
    socketio.run(app, debug=True)
