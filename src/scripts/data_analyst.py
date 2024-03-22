from chatbot import Chatbot
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',  # Log file path
    filemode='a',  # Append mode
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Define console colors
colors = {
    "YELLOW": "\033[33m",
    "GREEN": "\033[32m",
    "MAGENTA": "\033[35m",
    "BLUE": "\033[34m",
    "RESET": "\033[0m",
}

ANALYSIS_SYSTEM_PROMPT_FILEPATH = "prompts/data_analyst_system_direct_answer.txt"
INTENT_SYSTEM_PROMPT_FILEPATH = "prompts/data_analyst_system_intent_discovery.txt"
INTENT_USER_PROMPT_FILEPATH = "prompts/data_analyst_user_intent_discovery.txt"
ANALYSIS_USER_PROMPT_FILEPATH = "prompts/data_analyst_user_direct_answer.txt"

class PromptAnalyzer:
    def __init__(self):
        self.analysis_system_prompt = self.load_system_prompt(ANALYSIS_SYSTEM_PROMPT_FILEPATH)

        self.intent_system_prompt = self.load_system_prompt(INTENT_SYSTEM_PROMPT_FILEPATH)

        self.chatbot = Chatbot(
            model="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q8_0.gguf",
            base_url="http://localhost:1234/v1"
        )

    def _prepare_chatbot_for_analysis(self, conversation_history, system_prompt):
        # Clear existing conversation history
        self.chatbot.conversation.clear_history()

        # Set the appropriate system prompt
        self.chatbot.add_prompt_to_conversation("system", system_prompt)

        # Determine the start index for extracting exchanges
        start_index = max(len(conversation_history) - 5, 0)

        # Feed the last five exchanges from the user's conversation history
        for entry in conversation_history[start_index:]:
            if entry["role"] != "system":
                self.chatbot.add_prompt_to_conversation(entry["role"], entry["content"])

        formatted_history = ""

    def load_system_prompt(self, filename):
            """
            Load the system prompt from a file.

            Parameters:
            - filename (str): The name of the file containing the system prompt.

            Returns:
            - str: The content of the system prompt file.
            """
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    return file.read()
            except FileNotFoundError:
                print(f"Error: The file {filename} was not found.")
                return None

    def analyze(self, conversation_history:str) -> str:
        """
        Analyzes the conversation history to determine if the latest user prompt is directly answerable or requires further processing.

        Parameters:
            conversation_history (list[str]): A list of past user prompts.

        Returns:
            str: A decision indicating if the prompt is 'directly_answerable', 'requires_coding', 'requires_db_query', or 'requires_web_query'.
        """
        self._prepare_chatbot_for_analysis(conversation_history, self.analysis_system_prompt)
        # self.chatbot.add_prompt_to_conversation("system", self.analysis_system_prompt)

        # Encapsulate the conversation history in a prompt that aligns with the system prompt's mission
        analysis_prompt = self._prepare_analysis_prompt(conversation_history)

        # Use the embedded chatbot to generate a response based on the analysis prompt
        self.chatbot.add_prompt_to_conversation("user", analysis_prompt)
        decision = self.chatbot.generate_response()

        history = self.chatbot.conversation.history
        formatted_history = ""
        for entry in history:
            if entry['role'] == "system":
                formatted_history += f"{colors['GREEN']}{entry['role']}: {entry['content']}{colors['RESET']}\n"
            elif entry['role'] == "user":
                formatted_history += f"{colors['YELLOW']}{entry['role']}: {entry['content']}{colors['RESET']}\n"
            elif entry['role'] == "assistant":
                formatted_history += f"{colors['BLUE']}{entry['role']}: {entry['content']}{colors['RESET']}\n"

        logger.info(f"PromptAnalyzer.analyze():\n{formatted_history}")

        print(f"decision: {decision}")

        # Interpret the decision
        # print(f"PromptAnalyzer decision: {decision}")
        if decision.strip().lower() == "yes":
            # print("True")
            return True
        else:
            # print("False")
            return False

        # Simplify for demonstration. You would parse and interpret the chatbot's response here to return a meaningful decision.
        return decision

    def _prepare_analysis_prompt(self, conversation_history):
        """
        Prepares the analysis prompt by encapsulating the conversation history within instructions that reinforce the PromptAnalyzer's mission.

        Parameters:
            conversation_history (list[str]): The conversation history to encapsulate.

        Returns:
            str: The prepared analysis prompt.
        """
        # print(conversation_history)
        # Example: Combine the system prompt with the conversation history formatted as needed for analysis
#         formatted_history = "\n".join(entry['content'] for entry in conversation_history)
#
#         return f"{self.system_prompt}\n\nConversation history:\n{formatted_history}\n\nIs the latest question directly answerable? Provide a simple 'Yes' if the question is directly answerable by a language model like myself, or 'No' if it requires additional processing steps beyond natural language understanding."

        analysis_prompt = self.load_system_prompt(ANALYSIS_USER_PROMPT_FILEPATH)

        return analysis_prompt

    def analyze_intent(self, conversation_history) -> str:
        """
        Analyzes the query to determine its specific intent.

        Parameters:
            query (str): The user query to analyze.

        Returns:
            str: The determined intent of the query, such as 'requires_coding', 'requires_db_query', or 'requires_web_query'.
        """
        self._prepare_chatbot_for_analysis(conversation_history, self.intent_system_prompt)
        # Prepare the prompt for intent analysis
        intent_analysis_prompt = self._prepare_intent_analysis_prompt(conversation_history)

        # Use the chatbot to generate a response based on the intent analysis prompt
        self.chatbot.add_prompt_to_conversation("user", intent_analysis_prompt)
        intent_decision = self.chatbot.generate_response().strip().lower()

        history = self.chatbot.conversation.history
        formatted_history = ""
        formatted_history = ""
        for entry in history:
            if entry['role'] == "system":
                formatted_history += f"{colors['GREEN']}{entry['role']}: {entry['content']}{colors['RESET']}\n"
            elif entry['role'] == "user":
                formatted_history += f"{colors['YELLOW']}{entry['role']}: {entry['content']}{colors['RESET']}\n"
            elif entry['role'] == "assistant":
                formatted_history += f"{colors['BLUE']}{entry['role']}: {entry['content']}{colors['RESET']}\n"

        logger.info(f"PromptAnalyzer.analyze_intent():\n{formatted_history}")


        print(f"intent 1: {intent_decision}")

        return intent_decision

        # Map the model's response to specific actions
        # if intent_decision == "requires_coding":
        #     return "requires_coding"
        # elif intent_decision == "requires_db_query":
        #     return "requires_db_query"
        # elif intent_decision == "requires_web_query":
        #     return "requires_web_query"
        # else:
        #     return "undetermined"

    def _prepare_intent_analysis_prompt(self, conversation_history):
        # Here, you'd formulate a prompt that instructs the model to classify the query
        intent_prompt = self.load_system_prompt(INTENT_USER_PROMPT_FILEPATH)
        return intent_prompt

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
                try:
                    return self.chatbot.stream_response()
                    # yield from self.chatbot.stream_response()
                except Exception as e:
                    print(f"Error during streaming: {e}")
                    raise

                # return self.chatbot.stream_response()
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
        self.conversation_history = []

    def handle_prompt(self, user_input: str, streaming: bool = False):
        # Add the latest user input to the conversation history
        self.conversation_history = self.chatbot.conversation.history

        # User PromptAnalyzer to anlyze the conversation history
        answer_directly = self.prompt_analyzer.analyze(self.conversation_history)

        # Decision logic based on the analysis_result
        if answer_directly:
            return self.response_generator.generate_response(user_input, streaming)

        # if can_respond_directly:
        #     return self.response_generator.generate_response(user_input, streaming)
        else:
            pass
            intent = self.prompt_analyzer.analyze_intent(self.conversation_history)

            print(f"intent 2: {intent}")

            if intent == "requires_coding":
                return "This prompt requires coding.\n"
            elif intent == "requires_web_api_call":
                return "This prompt requires a web API call\n"
            elif intent == "requires_db_query":
                return "This prompt requires a database query\n"
            elif intent == "requires_web_query":
                return "This prompt requires a web query\n"
            else:
                print(f"intent 3: {intent}")
                return "Not sure what to do with this query. Please rephrase your question\n"

        # final_prompt = handle_user_input(user_input)

        # Optional Adjust final_prompt based on analysis_result

        # Generate response through ResponseGenerator, respecting the streaming setting

        # use prompt_analyzer to analyze the prompt and decide the necessary action (fetch data, generate code or answer directly)
        # based on the decision, call appropriate class
        # call response_generator
        # return self.response_generator.response(final_prompt, streaming)

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
