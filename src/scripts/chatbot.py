from abc import ABC, abstractmethod
import backoff
import requests
import tiktoken
import os
from huggingface_hub import HfApi
from transformers import AutoTokenizer, LlamaTokenizerFast

DEFAULT_TOKENIZER = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)

# Disable parallel tokenization to avoid potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    def estimate_token_count(self, model, text):
        """
        Calculate the maximum context window for a given model, typically defined by
        token limits. This method should return the maximum context size the model supports.
        """
        pass


class OpenAIClient(ChatClient):
    def __init__(self):
        try:
            from openai import OpenAI

            self.client = OpenAI()
            self.estimated_prompt_token_count = 0
        except Exception as e:
            self.client = None

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
        response_text = response.choices[0].message.content
        token_count = response.usage.total_tokens
        return response_text, token_count

    def stream_response(self, model, conversation_history):
        stream = self.client.chat.completions.create(
            model=model, messages=conversation_history, stream=True
        )
        full_response_text = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                # Append the chunk's content to the full response text
                full_response_text += content
                # Yield each chunk's content in real-time
                yield content

            # Check for the finish_reason to see if this is the last chunk
            if chunk.choices[0].finish_reason in ["stop", "length", "idle"]:
                # After the loop, once all chunks have been processed, including the final one
                break

        # The stream is finished; now calculate the total token count for the full response
        response_token_count = self.estimate_token_count(model, full_response_text)
        # Combine the response token count and the estimated prompt count
        token_count = response_token_count + self.estimated_prompt_token_count

        # Return or yield a special signal indicating the end of the stream along with the total token count
        yield None, token_count

    def estimate_token_count(self, model, text):
        """Estimates the token count for a given text using tiktoken."""
        try:
            # Use tiktoken to automatically find the correct encoding for the model
            encoding = tiktoken.encoding_for_model(model)
            # Assuming `encode` is the method to tokenize the text based on the encoding
            tokens = encoding.encode(text)
            token_count = len(tokens)
            return token_count
        except Exception as e:
            # Handle cases where tiktoken or encoding lookup fails
            print(f"Error estimating token count with tiktoken for model {model}: {e}")
            # Fallback to a simple approximation if necessary
            return len(text.split())  # Rough approximation


class OllamaClient(ChatClient):
    def __init__(self):
        try:
            import ollama

            self.client = ollama
            self.estimated_prompt_token_count = 0
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
        response_text = response["message"]["content"]

        # Extract the token counts for input and output
        input_token_count = response.get("prompt_eval_count", 0)
        output_token_count = response.get("eval_count", 0)

        # Combine the token counts as the total count for the interaction
        total_token_count = input_token_count + output_token_count

        return response_text, total_token_count

    def stream_response(self, model, conversation_history):
        stream = self.client.chat(
            model=model, messages=conversation_history, stream=True
        )
        for chunk in stream:
            if chunk["message"]["content"] is not None:
                yield chunk["message"]["content"]

            if chunk.get("done", False):
                # Break out of the loop if we're done
                break

        # After the stream is done, get the input and output token counts
        input_token_count = chunk.get("prompt_eval_count", 0)
        output_token_count = chunk.get("eval_count", 0)

        # Combine the token counts as teh total count for the interaction
        total_token_count = input_token_count + output_token_count

        # Yield a special signal indicating the end of the stream along with the total token count

        yield None, total_token_count

    def estimate_token_count(self, model, text):
        def fetch_base_model_identifier(model):
            """Attempt to fetch the base model identifier from the config.json file for a fine-tuned model."""
            config_url = f"https://huggingface.co/{model}/resolve/main/config.json"
            try:
                response = requests.get(config_url)
                response.raise_for_status()
                config_data = response.json()
                return config_data.get("_name_or_path")
            except requests.RequestException as e:
                print(f"Failed to fetch or parse config.json for {model}")

        def load_tokenizer_with_fallback(model):
            """Attempt to load a tokenizer, handling various errors and trying a fallback for fine-tuned models."""

            def handle_errors(e, model):
                if "404 Client Error" in str(e) or "Entry Not Found" in str(e):
                    print(
                        f"Tokenizer for {model} not found. Attempting to locate base model..."
                    )
                    base_model = fetch_base_model_identifier(model)
                    if base_model:
                        print(
                            f"Found base model {base_model}. Attempting to load its tokenizer"
                        )
                        return AutoTokenizer.from_pretrained(
                            base_model, trust_remote_code=True
                        )
                    else:
                        raise ValueError(
                            f"Base model identifier for {model} could not be found."
                        )
                elif "gated repo" in str(e).lower():
                    print(
                        f"Access to model {model} is restricted. See details below:\n{str(e)}\n"
                    )
                    print(f"Defaulting to LlamaTokenizerFast")
                    return DEFAULT_TOKENIZER
                else:
                    print(f"Unexpected error loading tokenizer for {model}: {e}")
                    print(f"Defaulting to LlamaTokenizerFast")
                    return DEFAULT_TOKENIZER

            try:
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            except Exception as e:
                print(str(e))
                tokenizer = handle_errors(e, model)

            return tokenizer

        hf_api = HfApi()

        try:
            models = hf_api.list_models(
                search=model, sort="likes", direction=-1, limit=1
            )
            model_found = next(models).id
            print(f"Model Identifier: {model_found}")

            tokenizer = load_tokenizer_with_fallback(model_found)

        except StopIteration:
            print(
                f"No models found for search term: {model}. Defaulting to LlamaTokenizerFast."
            )
            tokenizer = DEFAULT_TOKENIZER
        except Exception as e:
            print(
                f"An unexpected error occurred: {e}. Defaulting to LlamaTokenizerFast."
            )
            tokenizer = DEFAULT_TOKENIZER

        encoded_output = tokenizer.encode(text)
        estimated_token_count = len(encoded_output)
        return estimated_token_count


class Chatbot:
    def __init__(
        self,
        # system_prompt="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        system_prompt="You are a helpful assistant.",
        model="gpt-3.5-turbo",
        streaming=False,
    ):
        self.estimated_prompt_token_count = 0
        self.conversation_history_token_count = 0
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
        self.update_token_count_for_system_prompt()

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

    def estimate_token_count(self, content):
        # Ensure the client is initialized based on the current model
        if self.client:
            return self.client.estimate_token_count(self.model, content)
        else:
            return 0

    def generate_response(self):
        if not self.streaming:
            response_text, token_count = self.client.generate_response(
                self.model, self.conversation_history
            )
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
            self.conversation_history_token_count += token_count
            print(f"token count: {token_count}")
            print(
                f"updated conversation history count: {self.conversation_history_token_count}"
            )
            return response_text
        else:
            raise NotImplementedError(
                "Non-streaming mode is required for generate_response method."
            )

    def stream_response(self):
        if self.streaming:
            final_token_count = 0
            for item in self.client.stream_response(
                self.model, self.conversation_history
            ):
                if (
                    isinstance(item, tuple) and item[0] is None
                ):  # Check for the final token count
                    final_token_count = item[1]
                else:
                    response_text = item
                    self.conversation_history.append(
                        {"role": "assistant", "content": response_text}
                    )
                    yield response_text

            self.conversation_history_token_count += final_token_count
            print(f"final token count: {final_token_count}")
            print(
                f"updated conversation history count: {self.conversation_history_token_count}"
            )
        else:
            raise NotImplementedError(
                "Streaming mode is required for stream_response method."
            )

    def add_user_prompt(self, user_prompt):
        """Adds a user prompt to the conversation history and updates the token count."""
        self.conversation_history.append({"role": "user", "content": user_prompt})
        if self.client:
            # Correctly pass the model to estimate_token_count
            self.estimated_prompt_token_count = self.client.estimate_token_count(
                self.model, user_prompt
            )
            self.client.estimated_prompt_token_count = self.estimated_prompt_token_count

            print(
                f"Estimated token count of the user prompt: {self.estimated_prompt_token_count}"
            )

    def update_token_count_for_system_prompt(self):
        """Estimates token count for the system prompt and resets the conversation's token count."""
        if self.client:
            # Correctly pass the model to estimate_token_count
            self.estimated_prompt_token_count += self.client.estimate_token_count(
                self.model, self.system_prompt
            )
            print(
                f"estimated prompt token count of updated system prompt: {self.estimated_prompt_token_count}"
            )
            self.init_conversation()  # Reset or initialize the conversation history

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
