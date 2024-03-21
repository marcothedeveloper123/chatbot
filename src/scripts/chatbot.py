import contextlib
import io
import sys

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from transformers import AutoTokenizer, LlamaTokenizerFast
    from transformers import logging as token_logger

    token_logger.set_verbosity_error()

from abc import ABC, abstractmethod
import backoff
import requests
import tiktoken
import os
from huggingface_hub import HfApi
import time
import uuid

DEFAULT_TOKENIZER = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
DEFAULT_TEMPERATURE = 0.8
CLIENT_OPENAI = "openai"
CLIENT_OLLAMA = "ollama"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
STATE_SERVICE_UNAVAILABLE = "service_unavailable"
STATE_MODEL_UNAVAILABLE = "service_unavailable"
STATE_AVAILABLE = "available"

# Disable parallel tokenization to avoid potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Conversation:
    """
    Manages the conversation history and token counts for a chatbot session.

    Attributes:
        model (str): The model name being used for the conversation, if applicable.
    """

    model = None

    def __init__(self, max_token_count=8192):
        """
        Initializes a new conversation with an empty history and zero total token count.
        """
        self._history = []
        self.total_token_count = 0
        self.estimated_total_token_count = 0
        self.max_token_count = max_token_count

    def add_prompt(self, role, content, token_count=0, streaming=False):
        """
        Adds a prompt to the conversation history with its associated role, content, and token count.

        Parameters:
            role (str): The role of the prompt (e.g., 'user', 'system', 'assistant').
            content (str): The text content of the prompt.
            token_count (int): The estimated token count for the prompt.
        """
        # print(f"\n\n***<{role}> PROMPT TOKEN COUNT: {token_count}***\n\n")
        if streaming:
            self.total_token_count += token_count
            self._history.append(
                {
                    "role": role,
                    "content": content,
                    "token_count": self.total_token_count,
                    "estimated_total_token_count": None,
                }
            )
        else:
            self.estimated_total_token_count += token_count
            self._history.append(
                {
                    "role": role,
                    "content": content,
                    "token_count": None,
                    "estimated_total_token_count": self.estimated_total_token_count,
                }
            )
        self.manage_token_count()

    def add_response(self, client_name, response, streaming=False):
        """
        Adds a response from the chatbot to the conversation history and updates the total token count.

        Parameters:
            client_name (str): The name of the client providing the response (e.g., 'openai', 'ollama').
            response (dict): The response object from the client.
            streaming (bool): Indicates whether the response is from a streaming operation, affecting token count handling.
        """
        content, token_count = "", 0
        if client_name == CLIENT_OLLAMA:
            content = response["message"]["content"]
            token_count = response["prompt_eval_count"] + response["eval_count"]
        elif client_name == CLIENT_OPENAI:
            if streaming:
                content = response["choices"][0]["message"]["content"]
                token_count = response["usage"]["total_tokens"]
            else:
                content = response.choices[0].message.content
                token_count = response.usage.total_tokens
        else:
            raise ValueError("Unsupported client type.")

        # print(f"\n\n***<TOTAL TOKEN COUNT AS PER OPENAI: {token_count}***\n\n")

        if streaming:
            self.total_token_count += token_count
            self._history.append(
                {
                    "role": "assistant",
                    "content": content,
                    "token_count": self.total_token_count,
                    "estimated_total_token_count": None,
                }
            )
        else:
            self.total_token_count = token_count
            self.estimated_total_token_count = self.total_token_count
            self._history.append(
                {
                    "role": "assistant",
                    "content": content,
                    "token_count": self.total_token_count,
                    "estimated_total_token_count": self.estimated_total_token_count,
                }
            )

    def estimate_token_count(self, client_name, model, text):
        """
        Estimates the token count for a given text based on the specified client and model.

        Parameters:
            client_name (str): The name of the client (e.g., 'openai', 'ollama') used for estimating token count.
            model (str): The model being used for the conversation.
            text (str): The text content for which the token count is to be estimated.

        Returns:
            int: The estimated token count for the text.
        """
        if client_name == CLIENT_OPENAI:
            client = OpenAIClient()
        elif client_name == CLIENT_OLLAMA:
            client = OllamaClient()

        # Ensure the client is initialized based on the current model
        return client.estimate_token_count(model, text)

    def manage_token_count(self):
        # print(
        #     f"***MANAGING TOKEN COUNT: estimtated total = {self.estimated_total_token_count}, total = {self.total_token_count}, max = {self.max_token_count} ***"
        # )
        while (
            self.estimated_total_token_count > self.max_token_count - 1500
            and len(self._history) > 1
        ):
            # Remove the next oldest entry after the system prompt, ensuring the conversation stays within token limits
            self.pop(1)  # Adjusted to use the pop method for removal

    @property
    def history(self):
        """
        Returns the conversation history.

        Returns:
            list[dict]: A list of dictionaries, each representing a prompt or response with role, content, and token count.
        """
        return [
            {"role": entry["role"], "content": entry["content"]}
            for entry in self._history
        ]

    def history_log(self):
        return self._history

    @property
    def token_count(self):
        """
        Returns the total token count for the conversation.

        Returns:
            int: The total token count accumulated over the course of the conversation.
        """
        return self.total_token_count

    def set_history(self, new_history):

        # Adjust to ensure last entry is from 'assistant'
        if len(new_history) > 2:
            while new_history and new_history[-1]["role"] != "assistant":
                new_history.pop()

        # Set the new history and token counts
        self._history = new_history
        if len(self._history) > 2:
            self.total_token_count = new_history[-1][
                "token_count"
            ]  # the total_token_count for the last record in new_history
            self.estimated_total_token_count = new_history[-1][
                "estimated_total_token_count"
            ]  # the estimated_total_token_count for the last record in new_history

    def clear_history(self):
        self._history = []
        self.total_token_count = 0
        self.estimated_total_token_count = 0

    def pop(self, position):
        """
        Removes an entry from the conversation history at the specified position.

        Parameters:
            position (int): The index of the entry to be removed from the conversation history.

        Note: This method does not allow removing the initial system prompt (index 0).
        """
        # print(f"\n\n***POPPING MESSAGES:***\n\n")
        if position == 0:
            raise ValueError("Cannot remove the initial system prompt.")
        if position >= 1 and position < len(self._history):
            removed_entry = self._history.pop(position)
            self.total_token_count -= removed_entry["token_count"]
        else:
            raise IndexError("Position out of range.")


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

    @abstractmethod
    def estimate_token_count(self, model, text):
        """
        Calculate the maximum context window for a given model, typically defined by
        token limits. This method should return the maximum context size the model supports.
        """
        pass


class OpenAIClient(ChatClient):
    """
    A client class for interacting with the OpenAI API. This class handles listing available models, generating responses
    to conversation history, streaming responses for real-time interaction, and estimating token counts for given texts.

    Attributes:
        client (OpenAI): The OpenAI client initialized with the API key.
        estimated_prompt_token_count (int): A running total of the token count used for prompts.
        name (str): Identifier for the client, set to 'openai'.
        model (str, None): The AI model being used for generating responses. Set dynamically.
    """

    def __init__(self, base_url=""):
        """
        Initializes the OpenAI client and sets the base configuration.
        """
        try:
            from openai import OpenAI
            self.base_url = base_url

            if self.base_url == "":
                # print("openai!")
                self.client = OpenAI()
            else:
                # print("base url!")
                self.client = OpenAI(base_url=self.base_url, api_key="lm-studio")
            self.estimated_prompt_token_count = 0
            self.name = CLIENT_OPENAI
            self.model = None
            self._temperature = DEFAULT_TEMPERATURE
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            self.client = None

#     @property
#     def base_url(self):
#         return self._base_url
#
#     @base_url.setter
#     def base_url(self, value):
#         url = str(value)
#         self._base_url = url

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        temp = float(value)
        self._temperature = temp

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=3,
        giveup=lambda e: e.response is not None and e.response.status_code < 500,
    )
    def list_models(self):
        """
        Lists all available models from the OpenAI API.

        Returns:
            list[str]: A list of available model identifiers.
        """
        try:
            # print("listing models!!!")
            models = self.client.models.list()
            # print(f"models: {models}")
            # sys.exit()
            return [model.id for model in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def generate_response(self, conversation_history):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation_history,
            temperature=self._temperature,
        )
        """
        Generates a response from the AI model based on the given conversation history.

        Parameters:
            conversation_history (list[dict]): The conversation history with user and system prompts.

        Returns:
            tuple: A tuple containing the response text, the token count of the response, and the full response object.
        """
        response_text = response.choices[0].message.content
        token_count = response.usage.total_tokens
        return response_text, token_count, response

    def stream_response(self, conversation_history):
        """
        Streams the response from the AI model in real-time, allowing for incremental output as the model generates the response.

        Parameters:
            conversation_history (list[dict]): The conversation history with user and system prompts.

        Yields:
            str: Portions of the response text as they are generated by the AI model.

        After all chunks have been yielded, it yields a tuple containing None, the total token count, and a consolidated response object.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=conversation_history,
            stream=True,
            temperature=self._temperature,
        )
        full_response_text = ""
        finish_reason = None
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                # Append the chunk's content to the full response text
                full_response_text += content
                # Yield each chunk's content in real-time
                yield content

            # Check for the finish_reason to see if this is the last chunk
            if chunk.choices[0].finish_reason in ["stop", "length", "idle"]:
                finish_reason = chunk.choices[0].finish_reason
                # After the loop, once all chunks have been processed, including the final one
                break

        # The stream is finished; now calculate the total token count for the full response
        response_token_count = self.estimate_token_count(full_response_text)
        # Combine the response token count and the estimated prompt count
        token_count = response_token_count + self.estimated_prompt_token_count

        choices = [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": full_response_text.strip(),
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ]

        consolidated_response = {
            "id": f"chatcompl-{str(uuid.uuid4())}",  # Generate a unique ID for the session
            "choices": choices,
            "created": int(time.time()),
            "model": self.model,
            "object": "chat.completion",
            "system_fingerprint": f"fp_(str(uuid.uuid4()))",  # Generate a placeholder system fingerprint
            "usage": {
                "completion_tokens": response_token_count,
                "prompt_tokens": self.estimated_prompt_token_count,
                "total_tokens": token_count,
            },
        }

        # Return or yield a special signal indicating the end of the stream along with the total token count
        yield None, token_count, consolidated_response

    def estimate_token_count(self, text):
        """
        Estimates the token count of the given text using the model's tokenizer.

        Parameters:
            text (str): The text to be tokenized and counted.

        Returns:
            int: The estimated number of tokens in the text.
        """
        if self.base_url == "":
            try:
                # Use tiktoken to automatically find the correct encoding for the model
                encoding = tiktoken.encoding_for_model(self.model)
                # Assuming `encode` is the method to tokenize the text based on the encoding
                tokens = encoding.encode(text)
                token_count = len(tokens)
                return token_count
            except Exception as e:
                # Handle cases where tiktoken or encoding lookup fails
                print(
                    f"Error estimating token count with tiktoken for model {self.model}: {e}"
                )
                # Fallback to a simple approximation if necessary
                return len(text.split())  # Rough approximation
        else:
            """
            Estimates the token count of the given text. Utilizes a tokenizer that is compatible with the Ollama model to accurately count tokens.

            Parameters:
                text (str): The text to be tokenized and counted.

            Returns:
                int: The estimated number of tokens in the text.
            """

            def transform_model_name(model):
                """
                Transforms the model name to handle specific naming conventions, especially for models with versioning or specific configurations.

                Parameters:
                    model (str): The original model name.

                Returns:
                    str: The transformed model name suitable for searching or tokenization.
                """
                # Split the model name from its descriptor
                parts = model.split(":")
                model_name = parts[0]
                # If model descriptor is 'latest', return the model name as is
                if parts[-1] == "latest":
                    return model_name
                else:
                    # For other descriptors, split by '-' and take relevant parts
                    descriptor_parts = parts[1].split("-")
                    # Include 'chat' specific handling with refined logic
                    if "chat" in descriptor_parts:
                        # Find the position of 'chat' and include the segment after 'chat' if it's part of the descriptor
                        index_of_chat = descriptor_parts.index("chat")
                        relevant_parts = descriptor_parts[
                            : index_of_chat + 1
                        ]  # Include up to 'chat'
                        # Check if there's a version or identifier immediately after 'chat' and include it
                        if len(
                            descriptor_parts
                        ) > index_of_chat + 1 and not descriptor_parts[
                            index_of_chat + 1
                        ].endswith(
                            ("K_M", "fp16")
                        ):
                            relevant_parts.append(descriptor_parts[index_of_chat + 1])
                    else:
                        relevant_parts = descriptor_parts[:2]
                    # Reconstruct the model name with spaces and return
                    return " ".join([model_name] + relevant_parts)

            def fetch_base_model_identifier(model):
                """
                Fetches the base model identifier for a fine-tuned model by accessing its configuration file.

                Parameters:
                    model (str): The fine-tuned model name.

                Returns:
                    str: The identifier of the base model.
                """
                config_url = f"https://huggingface.co/{model}/resolve/main/config.json"
                try:
                    response = requests.get(config_url)
                    response.raise_for_status()
                    config_data = response.json()
                    return config_data.get("_name_or_path")
                except requests.RequestException as e:
                    # print(f"Failed to fetch or parse config.json for {model}")
                    raise

            def load_tokenizer_with_fallback(model):
                """
                Attempts to load a tokenizer for the given model, with fallback mechanisms for handling errors or restricted access.

                Parameters:
                    model (str): The model name.

                Returns:
                    A tokenizer instance compatible with the model.
                """

                def handle_errors(e, model):
                    """
                    Handles errors encountered when attempting to load a tokenizer for the specified model. It attempts to resolve
                    issues such as missing tokenizers for specific models, access restrictions, or any other unexpected errors
                    by applying fallback strategies.

                    Parameters:
                        e (Exception): The exception that was raised during the tokenizer loading attempt.
                        model (str): The model name for which the tokenizer loading was attempted.

                    Returns:
                        A tokenizer instance. This could either be the specific tokenizer for the model, a tokenizer for the base model
                        if the specific tokenizer is not found, or a default tokenizer if neither the specific nor base model tokenizer
                        can be loaded.

                    Raises:
                        ValueError: If no base model identifier can be found for the given model, indicating that the model might not
                        exist or there's an issue with its configuration on the hosting platform.
                    """
                    if "404 Client Error" in str(e) or "Entry Not Found" in str(e):
                        # print(
                        #     f"Tokenizer for {model} not found. Attempting to locate base model..."
                        # )
                        base_model = fetch_base_model_identifier(model)
                        if base_model:
                            # print(
                            #     f"Found base model {base_model}. Attempting to load its tokenizer"
                            # )
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
                        # print(f"Defaulting to LlamaTokenizerFast")
                        return DEFAULT_TOKENIZER
                    else:
                        # print(f"Unexpected error loading tokenizer for {model}: {e}")
                        # print(f"Defaulting to LlamaTokenizerFast")
                        return DEFAULT_TOKENIZER

                try:
                    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                except Exception as e:
                    # print(str(e))
                    tokenizer = handle_errors(e, model)

                return tokenizer

            hf_api = HfApi()

            try:
                model = transform_model_name(self.model)
                models = hf_api.list_models(
                    search=model, sort="likes", direction=-1, limit=1
                )
                model_found = next(models).id
                # print(f"Model Identifier: {model_found}")

                tokenizer = load_tokenizer_with_fallback(model_found)

            except StopIteration:
                # print(
                #     f"No models found for search term: {model}. Defaulting to LlamaTokenizerFast."
                # )
                tokenizer = DEFAULT_TOKENIZER
            except Exception as e:
                # print(
                #     f"An unexpected error occurred: {e}. Defaulting to LlamaTokenizerFast."
                # )
                tokenizer = DEFAULT_TOKENIZER

            encoded_output = tokenizer.encode(text)
            estimated_token_count = len(encoded_output)
            return estimated_token_count

class OllamaClient(ChatClient):
    """
    A client class for interacting with the Ollama API. This class is responsible for
    listing available models, generating responses based on the conversation history,
    streaming responses for real-time interaction, and estimating token counts for given texts.

    Attributes:
        client (Ollama): The Ollama client used to communicate with the Ollama API.
        estimated_prompt_token_count (int): Estimated token count of the prompt. Used for tracking token usage.
        name (str): Identifier for the client, set to 'CLIENT_OLLAMA'.
        model (str, None): The AI model being used for generating responses. This is set dynamically.
    """

    def __init__(self):
        """
        Initializes the Ollama client. Attempts to set up communication with the Ollama API.
        """
        try:
            import ollama

            self.client = ollama
            self.estimated_prompt_token_count = 0
            self.name = CLIENT_OLLAMA
            self.model = None
            self._temperature = DEFAULT_TEMPERATURE
        except Exception as e:
            self.client = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        temp = float(value)
        self._temperature = temp

    def list_models(self):
        """
        Lists all available models from the Ollama API.

        Returns:
            list[str]: A list of available model names.
        """
        if not self.client:
            return []  # Ollama not available
        try:
            models = self.client.list()
            return [model["name"] for model in models["models"]]
        except Exception as e:
            return []

    def generate_response(self, conversation_history):
        """
        Generates a response from the AI model based on the given conversation history.

        Parameters:
            conversation_history (list[dict]): The conversation history with user and system prompts.

        Returns:
            tuple: A tuple containing the response text, the total token count of the interaction, and the full response object.
        """
        response = self.client.chat(
            model=self.model,
            messages=conversation_history,
            options={"temperature": self._temperature},
        )
        response_text = response["message"]["content"]

        # Extract the token counts for input and output
        input_token_count = response.get("prompt_eval_count", 0)
        output_token_count = response.get("eval_count", 0)

        # Combine the token counts as the total count for the interaction
        total_token_count = input_token_count + output_token_count

        return response_text, total_token_count, response

    def stream_response(self, conversation_history):
        """
        Streams the response from the AI model in real-time, allowing for incremental output as the model generates the response.

        Parameters:
            conversation_history (list[dict]): The conversation history with user and system prompts.

        Yields:
            str: Portions of the response text as they are generated by the AI model.

        After all chunks have been yielded, it yields a tuple containing None, the total token count, and a consolidated response object that summarizes the entire streaming interaction.
        """
        prompt = conversation_history[-1]["content"]
        input_token_count = self.estimate_token_count(prompt)

        stream = self.client.chat(
            model=self.model,
            messages=conversation_history,
            stream=True,
            options={"temperature": self._temperature},
        )
        full_response_text = ""
        total_duration = load_duration = prompt_eval_duration = eval_duration = 0
        prompt_eval_count = eval_count = 0
        done = False

        for chunk in stream:
            content = chunk["message"]["content"]
            if content is not None:
                full_response_text += content
                yield chunk["message"]["content"]

            if chunk.get("done", False):
                done = chunk["done"]
                total_duration = chunk.get("total_duration", 0)
                load_duration = chunk.get("load_duration", 0)
                prompt_eval_duration = chunk.get("prompt_eval_duration", 0)
                eval_duration = chunk.get("eval_duration", 0)
                eval_count = chunk.get("eval_count", 0)
                # Break out of the loop if we're done
                break

        # After the stream is done, get the output token counts
        output_token_count = chunk.get("eval_count", 0)

        # Combine the token counts as the total count for the interaction
        total_token_count = input_token_count + output_token_count

        consolidated_response = {
            "model": self.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": full_response_text.strip()},
            "done": done,
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
        }

        # Yield a special signal indicating the end of the stream along with the total token count

        yield None, total_token_count, consolidated_response

    def estimate_token_count(self, text):
        """
        Estimates the token count of the given text. Utilizes a tokenizer that is compatible with the Ollama model to accurately count tokens.

        Parameters:
            text (str): The text to be tokenized and counted.

        Returns:
            int: The estimated number of tokens in the text.
        """

        def transform_model_name(model):
            """
            Transforms the model name to handle specific naming conventions, especially for models with versioning or specific configurations.

            Parameters:
                model (str): The original model name.

            Returns:
                str: The transformed model name suitable for searching or tokenization.
            """
            # Split the model name from its descriptor
            parts = model.split(":")
            model_name = parts[0]
            # If model descriptor is 'latest', return the model name as is
            if parts[-1] == "latest":
                return model_name
            else:
                # For other descriptors, split by '-' and take relevant parts
                descriptor_parts = parts[1].split("-")
                # Include 'chat' specific handling with refined logic
                if "chat" in descriptor_parts:
                    # Find the position of 'chat' and include the segment after 'chat' if it's part of the descriptor
                    index_of_chat = descriptor_parts.index("chat")
                    relevant_parts = descriptor_parts[
                        : index_of_chat + 1
                    ]  # Include up to 'chat'
                    # Check if there's a version or identifier immediately after 'chat' and include it
                    if len(
                        descriptor_parts
                    ) > index_of_chat + 1 and not descriptor_parts[
                        index_of_chat + 1
                    ].endswith(
                        ("K_M", "fp16")
                    ):
                        relevant_parts.append(descriptor_parts[index_of_chat + 1])
                else:
                    relevant_parts = descriptor_parts[:2]
                # Reconstruct the model name with spaces and return
                return " ".join([model_name] + relevant_parts)

        def fetch_base_model_identifier(model):
            """
            Fetches the base model identifier for a fine-tuned model by accessing its configuration file.

            Parameters:
                model (str): The fine-tuned model name.

            Returns:
                str: The identifier of the base model.
            """
            config_url = f"https://huggingface.co/{model}/resolve/main/config.json"
            try:
                response = requests.get(config_url)
                response.raise_for_status()
                config_data = response.json()
                return config_data.get("_name_or_path")
            except requests.RequestException as e:
                # print(f"Failed to fetch or parse config.json for {model}")
                raise

        def load_tokenizer_with_fallback(model):
            """
            Attempts to load a tokenizer for the given model, with fallback mechanisms for handling errors or restricted access.

            Parameters:
                model (str): The model name.

            Returns:
                A tokenizer instance compatible with the model.
            """

            def handle_errors(e, model):
                """
                Handles errors encountered when attempting to load a tokenizer for the specified model. It attempts to resolve
                issues such as missing tokenizers for specific models, access restrictions, or any other unexpected errors
                by applying fallback strategies.

                Parameters:
                    e (Exception): The exception that was raised during the tokenizer loading attempt.
                    model (str): The model name for which the tokenizer loading was attempted.

                Returns:
                    A tokenizer instance. This could either be the specific tokenizer for the model, a tokenizer for the base model
                    if the specific tokenizer is not found, or a default tokenizer if neither the specific nor base model tokenizer
                    can be loaded.

                Raises:
                    ValueError: If no base model identifier can be found for the given model, indicating that the model might not
                    exist or there's an issue with its configuration on the hosting platform.
                """
                if "404 Client Error" in str(e) or "Entry Not Found" in str(e):
                    # print(
                    #     f"Tokenizer for {model} not found. Attempting to locate base model..."
                    # )
                    base_model = fetch_base_model_identifier(model)
                    if base_model:
                        # print(
                        #     f"Found base model {base_model}. Attempting to load its tokenizer"
                        # )
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
                    # print(f"Defaulting to LlamaTokenizerFast")
                    return DEFAULT_TOKENIZER
                else:
                    # print(f"Unexpected error loading tokenizer for {model}: {e}")
                    # print(f"Defaulting to LlamaTokenizerFast")
                    return DEFAULT_TOKENIZER

            try:
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            except Exception as e:
                # print(str(e))
                tokenizer = handle_errors(e, model)

            return tokenizer

        hf_api = HfApi()

        try:
            model = transform_model_name(self.model)
            models = hf_api.list_models(
                search=model, sort="likes", direction=-1, limit=1
            )
            model_found = next(models).id
            # print(f"Model Identifier: {model_found}")

            tokenizer = load_tokenizer_with_fallback(model_found)

        except StopIteration:
            # print(
            #     f"No models found for search term: {model}. Defaulting to LlamaTokenizerFast."
            # )
            tokenizer = DEFAULT_TOKENIZER
        except Exception as e:
            # print(
            #     f"An unexpected error occurred: {e}. Defaulting to LlamaTokenizerFast."
            # )
            tokenizer = DEFAULT_TOKENIZER

        encoded_output = tokenizer.encode(text)
        estimated_token_count = len(encoded_output)
        return estimated_token_count


class Chatbot:
    """
    Orchestrates the conversation flow between a user and an AI model, managing prompts, responses, and token counts.

    This class integrates with AI client services (e.g., OpenAI, Ollama) to generate responses to user prompts and manage the conversation history. It also handles token counting for estimating the conversation's complexity and ensures compatibility with the model's context window limits.

    Attributes:
        system_prompt (str): A predefined message that sets the context or personality of the AI model at the start of the conversation.
        model (str): The identifier of the AI model used for generating responses. Default is "gpt-3.5-turbo".
        streaming (bool): Indicates whether the chatbot should use streaming mode for generating responses. Default is False.
    """

    def __init__(
        self,
        system_prompt="You are a helpful assistant.",
        base_url="",
        model="gpt-3.5-turbo",
        streaming=False,
        max_token_count=8192,
        conversation_history=None,
        temperature=DEFAULT_TEMPERATURE,
    ):
        """
        Initializes a Chatbot instance with a specified system prompt, model, and streaming mode.

        Parameters:
        - system_prompt: A string representing the initial message or context for the chatbot.
        - model: A string indicating the model to be used for generating responses.
        - streaming: A boolean flag to determine if the responses should be streamed.
        """
        self.client = None
        self.base_url = base_url
        self._model = model
        self.streaming = streaming
        self._model_cache = {}
        self._initial_state = "initializing"
        self._temperature = temperature

        self.initialize_chatbot()

        if self._initial_state == STATE_AVAILABLE:
            self.system_prompt = system_prompt
            self.conversation = Conversation(max_token_count=max_token_count)
            if conversation_history:
                self.conversation.set_history(conversation_history)
            else:
                self.estimated_prompt_token_count = self.add_prompt_to_conversation(
                    ROLE_SYSTEM, self.system_prompt
                )
            self.conversation_history_token_count = self.conversation.total_token_count

    def initialize_chatbot(self):
        """
        Initializes the chatbot by determining which client supports the specified model,
        setting the client, and updating the chatbot's initial state based on client and model availability.
        """
        # Determine which client supports the specified model
        clients = {"openai": OpenAIClient(base_url = self.base_url), "ollama": OllamaClient()}

        clients_available = False
        model_supported = False

        for client_name, client in clients.items():
            try:
                # self.client = client
                # if client_name == "openai" and self.base_url != "":
                #     client.base_url = self.base_url
                available_models = client.list_models()
                self._model_cache[client_name] = available_models
                clients_available = True

                if self.model in available_models:
                    self.client = client
                    self.client.model = self.model
                    self.client.temperature = self._temperature
                    model_supported = True
            except Exception as e:
                self._model_cache[client_name] = []
                print(f"Error with {client_name} client: {e}")

        if not clients_available:
            self._initial_state = "service_unavailable"
        elif not model_supported:
            self._initial_state = "model_not_available"
        else:
            self._initial_state = "available"

    def add_prompt_to_conversation(self, role, content):
        """
        Adds a prompt to the conversation with an estimated token count.

        Parameters:
        - role: The role of the prompt ('user' or 'system').
        - content: The text content of the prompt.
        Returns:
        - The token count for the added prompt.
        """
        # print(f"***content:***{content}")
        token_count = self.client.estimate_token_count(content)
        self.conversation.add_prompt(role, content, token_count, self.streaming)

        if role == ROLE_SYSTEM:
            self.system_prompt = content

        return token_count

    @property
    def model(self):
        """
        A property decorator used to get the current model in use.
        """

        return self._model

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def model_cache(self):
        return self._model_cache

    @model.setter
    def model(self, value):
        """
        A setter for the model property that updates the model and initializes the client
        based on the new model if it's available in the cache.
        """

        if self._model != value:
            self._model = value
            self.initialize_chatbot

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        temp = float(value)
        self._temperature = temp

    def init_conversation(self, user_prompt=None):
        """
        Initializes the conversation history with the system prompt and optionally a user prompt.

        Parameters:
        - user_prompt: An optional initial user prompt to add to the conversation history.
        """
        if not self.conversation_history or (
            self.conversation_history
            and self.conversation_history[0]["content"] != self.system_prompt
        ):
            self.conversation_history.append(
                {"role": "system", "content": self.system_prompt}
            )
        if user_prompt:
            self.conversation_history.append({"role": "user", "content": user_prompt})

    def estimate_token_count(self, content):
        """
        Estimates the token count of a given piece of content using the current client's model.

        Parameters:
        - content: The text content for which to estimate the token count.
        Returns:
        - An integer representing the estimated number of tokens.
        """
        # Ensure the client is initialized based on the current model
        if self.client:
            return self.client.estimate_token_count(content)
        else:
            return 0

    def generate_response(self):
        """
        Generates a response from the chatbot based on the current conversation history.
        This method is intended for non-streaming mode.

        Returns:
        - A string containing the chatbot's response.
        """
        if not self.streaming:
            response_text, token_count, response = self.client.generate_response(
                self.conversation.history
            )
            self.conversation.add_response(self.client.name, response)
            return response_text
        else:
            raise NotImplementedError(
                "Non-streaming mode is required for generate_response method."
            )

    def stream_response(self):
        """
        Streams responses from the chatbot based on the current conversation history.
        This method is intended for streaming mode and yields responses incrementally.

        Yields:
        - Incremental parts of the chatbot's response as they become available.
        """
        if self.streaming:
            full_response_text = ""
            final_token_count = 0
            for item in self.client.stream_response(self.conversation.history):
                if (
                    isinstance(item, tuple) and item[0] is None
                ):  # Check for the final token count
                    final_token_count = item[1]
                    response = item[2]
                else:
                    response_text = item
                    full_response_text += response_text
                    yield response_text

            self.conversation.add_response(self.client.name, response, streaming=True)

            yield "\n"
        else:
            raise NotImplementedError(
                "Streaming mode is required for stream_response method."
            )

    # def add_user_prompt(self, user_prompt):
    #     """Adds a user prompt to the conversation history and updates the token count."""
    #     self.conversation_history.append({"role": "user", "content": user_prompt})
    #     if self.client:
    #         # Correctly pass the model to estimate_token_count
    #         self.estimated_prompt_token_count = self.client.estimate_token_count(
    #             self.model, user_prompt
    #         )
    #         self.client.estimated_prompt_token_count = self.estimated_prompt_token_count

    # def update_token_count_for_system_prompt(self):
    #     """Estimates token count for the system prompt and resets the conversation's token count."""
    #     if self.client:
    #         # Correctly pass the model to estimate_token_count
    #         self.estimated_prompt_token_count += self.client.estimate_token_count(
    #             self.model, self.system_prompt
    #         )
    #         self.init_conversation()  # Reset or initialize the conversation history

    # def update_system_prompt(self, new_prompt):
    #     """
    #     Enables the user to change the system prompt. We do this by replacing the system prompt
    #     at the very start of the conversation, as well as appending it to the end of the conversation.
    #     """
    #     self.system_prompt = new_prompt
    #     self.conversation_history[0] = {"role": "system", "content": self.system_prompt}
    #     self.conversation_history.append(
    #         {"role": "system", "content": self.system_prompt}
    #     )
