# Conversation history

## Documentation

The `conversation_history` within this chatbot framework serves as a central record of the interactions between a user and the chatbot, tracking both prompts and responses along with their associated token counts. This documentation outlines its format, evolution over time, dependencies, and usage.

### Format

The `conversation_history` is a list of dictionaries, where each dictionary represents a single entry in the conversation. Each entry contains the following keys:

- **`role`**: Identifies the source of the entry. Values can be `"user"`, `"assistant"`, or `"system"`, indicating whether the message originated from the user, the chatbot, or was a system-generated message, respectively.
- **`content`**: The text of the message or prompt.
- **`token_count`**: The number of tokens associated with the entry. For prompts, this is an estimated count, and for responses, it can be an actual count in non-streaming mode or an estimated count in streaming mode.
- **`total_token_count`** (proposed addition): The cumulative total of tokens used in the conversation up to this point, updated with actual token counts as responses are processed.
- **`estimated_total_token_count`** (proposed addition): Tracks the estimated cumulative token count, which is particularly relevant before actual response token counts are available or in streaming scenarios where token counts are incrementally updated.

### Evolution Over Time

`conversation_history` starts empty and grows as the chatbot session progresses. Each user prompt, assistant response, or system message is appended to the history in the order they occur, providing a linear record of the conversation. The `token_count` for each entry is calculated based on the text content, with the `total_token_count` and `estimated_total_token_count` being updated to reflect the evolving conversation's token usage.

### Dependencies

- **Tokenization and Token Count Estimation**: The `AutoTokenizer` from the Transformers library, or a custom tokenizer like `LlamaTokenizerFast`, is used to estimate the token count for each entry based on the model's vocabulary. The actual token usage for responses can also depend on the specifics of the underlying language model (LM) being used (e.g., OpenAI's GPT or Ollama).
- **Client Classes (`OpenAIClient`, `OllamaClient`)**: These classes interface with the respective APIs to generate responses based on the conversation history, contributing to the `conversation_history` with their outputs.

### Usage

- **Generating Responses**: `conversation_history` is passed to the client classes when generating responses, allowing the chatbot to consider the context of the conversation thus far. This enables more coherent and contextually relevant responses from the chatbot.
- **Token Count Management**: It plays a critical role in managing the conversation within the token limits of the underlying LM. By tracking the `total_token_count` and `estimated_total_token_count`, the framework can make informed decisions about pruning the conversation history or managing the conversation's complexity to stay within limits.
- **Conversation Analysis and Debugging**: Developers can analyze `conversation_history` to understand the flow of the conversation, identify issues, or gain insights into user interactions with the chatbot.

### Conclusion

`conversation_history` is a foundational element of the chatbot framework, facilitating coherent conversation flow, enabling context-aware responses, and ensuring compliance with token usage constraints. Its structured format, alongside the proposed additions of `total_token_count` and `estimated_total_token_count`, provides a comprehensive view of the conversation's dynamics, serving as a critical resource for both operational management and analytical purposes.
