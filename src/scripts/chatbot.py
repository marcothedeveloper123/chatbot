from abc import ABC, abstractmethod
import backoff
import requests


class ChatClient(ABC):
    """
    This is an abstract class, a kind of template for LLM client services such as
    OpenAI and Ollama. Every service will have its own implementation of this
    template.
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

    @abstractmethod
    def calculate_context_window(self, model):
        """
        Calculate the maximum context window for a given model, typically defined by
        token limits. This method should return the maximum context size the model supports.
        """
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
    def __init__(
        self,
        # system_prompt="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        system_prompt="You are a helpful assistant.",
        model="gpt-3.5-turbo",
        streaming=False,
    ):
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
        self.models_cache["openai"] = OpenAIClient().list_models()
        try:
            self.models_cache["ollama"] = OllamaClient().list_models()
        except Exception as e:
            self.models_cache["ollama"] = []

        self.update_initial_state()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
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
