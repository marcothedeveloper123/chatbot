"""
This module defines a framework for interacting with different Large Language Model (LLM) chat services,
such as OpenAI's GPT models and Ollama, through a unified interface. It abstracts the complexities of
directly interacting with these services' APIs, providing a simplified way to integrate LLMs into applications.

The framework consists of an abstract base class, `ChatClient`, which outlines the necessary methods any chat
service client should implement. Concrete classes `OpenAIClient` and `OllamaClient` provide specific implementations
for interacting with their respective services. The `Chatbot` class acts as a high-level interface, allowing users
to easily switch between different LLM services, manage conversation history, and customize the chatbot's behavior.

Example Implementation:
-----------------------

The following example demonstrates initializing a Chatbot instance, setting it up to use OpenAI's GPT-3 model,
sending a user prompt, and receiving a response.

```python
# Initialize the chatbot with a specific system prompt and model
chatbot = Chatbot(system_prompt="You are a helpful assistant.", model="gpt-3.5-turbo")

# Add a user prompt to the conversation
chatbot.add_user_prompt("Tell me a joke about programming.")

# Generate a response from the chatbot
response = chatbot.generate_response()

print(response)

This example sets up a basic interaction with the chatbot, demonstrating how to initialize the chatbot, add a user prompt, and receive a response. It's a starting point for integrating the chatbot into your projects, with flexibility to switch models or LLM services as needed.
"""

from abc import ABC, abstractmethod  # create abstract base classes
import backoff  # retry requests upon failures
import requests  # http requests


class ChatClient(ABC):
    """
    Abstract base class for chat service clients. It defines a common interface for all chat clients,
    ensuring that each client implementation, such as for OpenAI and Ollama, provides specific methods
    for listing models, generating responses, and streaming responses.
    """

    @abstractmethod
    def list_models(self):  # what models does the service offer?
        pass

    @abstractmethod
    def generate_response(self, model, conversation_history):
        """Generate a non-streaming response"""
        pass

    @abstractmethod
    def stream_response(self, model, conversation_history):  #
        """Generate a streaming response"""
        pass


class OpenAIClient(ChatClient):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=3,
        giveup=lambda e: e.response is not None and e.response.status_code < 500,
    )
    def list_models(self):
        try:
            models = self.client.models.list()
            return [model.id for model in models]
        except Exception as e:
            return []

    def generate_response(self, model, conversation_history):
        response = self.client.chat.completions.create(
            model=model, messages=conversation_history
        )
        return response.choices[0].message.content

    def stream_response(self, model, conversation_history):
        stream = self.client.chat.completions.create(
            model=model, messages=conversation_history, stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class OllamaClient(ChatClient):
    def __init__(self):
        try:
            import ollama

            self.client = ollama
        except Exception as e:
            self.client = None

    def list_models(self):
        if not self.client:
            return []  # Ollama not available
        try:
            models = self.client.list()
            return [model["name"] for model in models["models"]]
        except Exception as e:
            return []

    def generate_response(self, model, conversation_history):
        response = self.client.chat(model=model, messages=conversation_history)
        return response["message"]["content"]

    def stream_response(self, model, conversation_history):
        stream = self.client.chat(
            model=model, messages=conversation_history, stream=True
        )
        for chunk in stream:
            if chunk["message"]["content"] is not None:
                yield chunk["message"]["content"]


class Chatbot:
    """
    A chatbot class that abstracts the interaction with different LLM (Large Language Models) services
    through a unified interface. It supports initializing with a system prompt, choosing a model,
    and managing conversation history. It can interact with services by generating or streaming responses.
    """

    def __init__(
        self,
        # system_prompt="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        system_prompt="You are a helpful assistant.",
        model="gpt-3.5-turbo",
        streaming=False,
    ):
        """
        Initializes the chatbot with a default system prompt, model, and mode (streaming or non-streaming).
        It sets up the conversation history and initializes the models cache.
        """

        self.system_prompt = system_prompt
        self.streaming = streaming
        self.conversation_history = (
            []
        )  # TO DO deal with context window (model dependent!)
        # self.history = InMemoryHistory()
        self._model = None
        self.models_cache = {}
        self.client = None
        self._initial_state = "initializing"
        self.init_models_cache()
        # self.init_client()
        self.model = model
        self.init_conversation()

    def init_models_cache(self):
        """
        Initializes the models cache by listing available models from both OpenAI and Ollama clients.
        Updates the chatbot's initial state based on the availability of models.
        """

        self.models_cache["openai"] = OpenAIClient().list_models()
        try:
            self.models_cache["ollama"] = OllamaClient().list_models()
        except Exception as e:
            self.models_cache["ollama"] = []

        self.update_initial_state()

    @property
    def model(self):
        """
        A property decorator used to get the current model in use.
        """

        return self._model

    @model.setter
    def model(self, value):
        """
        A setter for the model property that updates the model and initializes the client
        based on the new model if it's available in the cache.
        """

        if self._model != value:
            self._model = value
            self.update_initial_state()
            if self._initial_state == "available":
                self.init_client()

    @property
    def initial_state(self):
        return self._initial_state

    def update_initial_state(self):
        if not self.models_cache["openai"] and not self.models_cache["ollama"]:
            self._initial_state = "service_unavailable"
        elif self._model in self.models_cache.get(
            "openai", []
        ) or self._model in self.models_cache.get("ollama", []):
            self._initial_state = "available"
        else:
            self._initial_state = "model_not_available"

    def init_client(self):
        """
        Initializes the appropriate client (OpenAI or Ollama) based on the current model setting.
        """

        if self._model:
            if self._model in self.models_cache["openai"]:
                self.client = OpenAIClient()
            elif self._model in self.models_cache["ollama"]:
                self.client = OllamaClient()
            else:
                raise ValueError(
                    "The specified model is not available in OpenAI or Ollama models."
                )

    def init_conversation(self, user_prompt=None):
        self.conversation_history.append(
            {"role": "system", "content": self.system_prompt}
        )
        if user_prompt:
            self.conversation_history.append({"role": "user", "content": user_prompt})

    def generate_response(self):
        """
        Generates a non-streaming response from the current client and updates the conversation history.
        Raises NotImplementedError if called in streaming mode.
        """

        if not self.streaming:
            response_text = self.client.generate_response(
                self.model, self.conversation_history
            )
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
            return response_text
        else:
            raise NotImplementedError(
                "Non-streaming mode is required for generate_response method."
            )

    def stream_response(self):
        """
        Yields a streaming response from the current client, updating the conversation history as responses are received.
        Raises NotImplementedError if not in streaming mode.
        """

        if self.streaming:
            for response_text in self.client.stream_response(
                self.model, self.conversation_history
            ):
                self.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )
                yield response_text
        else:
            raise NotImplementedError(
                "Streaming mode is required for stream_response method."
            )

    def add_user_prompt(self, user_prompt):
        self.conversation_history.append({"role": "user", "content": user_prompt})

    def update_system_prompt(self, new_prompt):
        """
        Enables the user to change the system prompt. We do this by replacing the system prompt
        at the very start of the conversation, as well as appending it to the end of the conversation.
        """
        self.system_prompt = new_prompt
        self.conversation_history[0] = {"role": "system", "content": self.system_prompt}
        self.conversation_history.append(
            {"role": "system", "content": self.system_prompt}
        )
