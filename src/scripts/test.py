import os
import requests
from huggingface_hub import HfApi
from transformers import AutoTokenizer, LlamaTokenizerFast

DEFAULT_TOKENIZER = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)

# Disable parallel tokenization to avoid potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def estimate_token_count(model, text):
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
        models = hf_api.list_models(search=model, sort="likes", direction=-1, limit=1)
        model_found = next(models).id
        print(f"Model Identifier: {model_found}")

        tokenizer = load_tokenizer_with_fallback(model_found)

    except StopIteration:
        print(
            f"No models found for search term: {model}. Defaulting to LlamaTokenizerFast."
        )
        tokenizer = DEFAULT_TOKENIZER
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Defaulting to LlamaTokenizerFast.")
        tokenizer = DEFAULT_TOKENIZER

    encoded_output = tokenizer.encode(text)
    estimated_token_count = len(encoded_output)
    return estimated_token_count


# name = "llama-2"
# name = "llama-pro"
# name = "mistral 7b"
name = "openhermes"
# name = "phi"
# name = "qwen"
# name = "stable-code"
# name = "stablelm2"

text = "Hello world"
token_count = estimate_token_count(name, text)
print(f"Estimated token count: {token_count}")
