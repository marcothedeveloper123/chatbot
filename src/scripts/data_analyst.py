from chatbot import Chatbot

class PromptAnalyzer:
    def __init__(self):
        pass

    def analyze(self, prompt: str) -> str:
        # Placeholder for analysis logic
        # For now, always return True indicating all prompts can be directly answered
        return True

class ExternalDataFetcher:
    def __init__(self):
        pass

    def fetch(self, query: str) -> str:
        pass

class CodeGenerator:
    def __init__(self):
        pass

    def generate_code(self, brief: str) -> str:
        pass

class ResponseGenerator:
        def __init__(self, chatbot):
            # Initialize with an instance of Chatbot
            self.chatbot = chatbot

        def generate_response(self, prompt: str, streaming: bool = False):
            # Assuming the prompt has been added to the chatbot's conversation history beforehand

            if streaming:
                # In streaming mode, yield each response part incrementally
                return self.chatbot.stream_response()
            else:
                # In non-streaming mode, return the complete response directly
                response_text = self.chatbot.generate_response()
                def single_response_iterable():
                    yield response_text
                return single_response_iterable()

class ConversationManager:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.prompt_analyzer = PromptAnalyzer()
        self.data_fetcher = ExternalDataFetcher()
        self.code_generator = CodeGenerator()
        self.response_generator = ResponseGenerator(self.chatbot)

    def handle_prompt(self, user_input: str, streaming: bool = False):
        can_respond_directly = self.prompt_analyzer.analyze(user_input)

        if can_respond_directly:
            return self.response_generator.generate_response(user_input, streaming)
        else:
            # Placeholder for alternative actions for prompts requiring external data or code execution
            # For now, let's just return a placeholder response indicating further action is needed
            # This branch can be expanded in the future to handle more complex scenarios
            if streaming:
                def placeholder_streaming_response():
                    yield "This prompt requires further processing. Streaming mode."
                return placeholder_streaming_response()
            else:
                return "This prompt requires further processing. Non-streaming mode."

        # final_prompt = handle_user_input(user_input)

        # Optional Adjust final_prompt based on analysis_result

        # Generate response through ResponseGenerator, respecting the streaming setting

        # use prompt_analyzer to analyze the prompt and decide the necessary action (fetch data, generate code or answer directly)
        # based on the decision, call appropriate class
        # call response_generator
        return self.response_generator.response(final_prompt, streaming)

    def add_prompt_to_conversation(self, role: str, prompt: str) -> int:
        """
        Adds a prompt to the conversation history, tagging it with the role of the participant (system or user).

        Parameters:
        - role: A string indiciating whether the prompt is from the `system` or `user` .__abs__()
        - prompt: The prompt or response text to be added to the conversation history.__abs__()

        Returns:
        - The number of tokens in the added prompt.
        """
        return self.chatbot.add_prompt_to_conversation(role, prompt)
