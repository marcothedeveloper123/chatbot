import os
import shutil
from datetime import datetime
import chromadb
from chromadb.config import Settings
import openai
import yaml
import json
from time import time, sleep
from uuid import uuid4
from chatbot import Chatbot

# LLM = "openhermes:latest"
# LLM = "samantha-mistral:7b-v1.2-text-fp16"
# LLM = "llama-pro:8b-instruct-q5_K_M"
LLM_main = "dolphin-mixtral:8x7b-v2.5-q5_1"
LLM_profile = LLM_main
LLM_kb = LLM_main

CONVERSATION_HISTORY_PATH = "history_logs/conversation_history.json"
USER_PROFILE_PATH = "user_profile.txt"
BACKUP_DIRECTORY = "profile_logs/"
DEFAULT_PROFILE_PATH = "user_profile_default.txt"
PROFILE_TEMPLATE_PATH = "user_profile_template.txt"
PROFILE_TEMPLATE_TEXT = """
- Name: UPDATE WITH YOUR NAME
- Profession: UPDATE WITH YOUR JORB
- Interests: UPDATE WITH YOUR HOBBIES
- Beliefs: WHAT DO YOU BELIEVE?
- Plans: WHAT ARE YOUR GOALS?
- Preference: HOW DO YOU PREFER TO COMMUNICATE?
"""

MAX_BACKUPS = 5
RETRY_ATTEMPTS = 3


def multi_line_input(prompt):
    initial_input = input(prompt)

    # Check if multi-line input is initiated
    if initial_input.endswith('"""'):
        print(">>>")  # Start of multi-line input
        input_lines = (
            [initial_input[:-3]] if initial_input[:-3].strip() else []
        )  # Exclude the trailing """

        while True:
            line = input()
            if line.strip() == '"""':
                print("<<<")  # End of multi-line input
                break
            input_lines.append(line)

        # After exiting multi-line, allow for an additional single line without automatic prompt.
        # Use an empty input to capture this line, showing the continuation of input but without a predefined prompt.
        final_line = input()  # No prompt for the final line
        if final_line.strip():
            input_lines.append(final_line)

        return "\n".join(input_lines)
    else:
        # Single line input that did not initiate multi-line mode
        return initial_input


def save_yaml(filepath, data):
    with open(filepath, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)


def save_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write(content)


def open_file(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
        return infile.read()


def chatbot(chatbot_instance, user_input):
    try:
        chatbot_instance.add_prompt_to_conversation("user", user_input)
        response_text = chatbot_instance.generate_response()

        # Prepare and log the conversation
        debug_object = [
            {"role": i["role"], "content": i["content"]}
            for i in chatbot_instance.conversation.history
        ]
        save_yaml("api_logs/convo_%s.yaml" % time(), debug_object)

        return response_text
    except Exception as e:
        print(f"\n\nError in chatbot function: '{e}'")


def backup_current_profile():
    """Backs up the current user profile with a timestamp."""
    if not os.path.exists(BACKUP_DIRECTORY):
        os.makedirs(BACKUP_DIRECTORY)
    if os.path.exists(USER_PROFILE_PATH) and os.path.getsize(USER_PROFILE_PATH) > 0:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = f"user_profile_backup_{timestamp}.txt"
        backup_path = os.path.join(BACKUP_DIRECTORY, backup_filename)
        shutil.copy(USER_PROFILE_PATH, backup_path)


def ensure_valid_user_profile():
    """
    Ensure the user profile is valid and not empty.
    If it's empty, try to recover from the latest backup.
    If no valid backup is found, use a default profile or prompt the user.
    """
    if not is_profile_valid(USER_PROFILE_PATH):
        recover_user_profile()
        if not is_profile_valid(USER_PROFILE_PATH):
            prompt_user_to_create_profile()


def is_profile_valid(profile_path):
    """Check if the profile file exists and is not empty."""
    return os.path.exists(profile_path) and os.path.getsize(profile_path) > 0


def recover_user_profile():
    """Attempt to recover the user profile from the latest backup."""
    backups = sorted(
        [f for f in os.listdir(BACKUP_DIRECTORY) if f.endswith(".txt")], reverse=True
    )
    for backup_file in backups:
        backup_path = os.path.join(BACKUP_DIRECTORY, backup_file)
        if is_profile_valid(backup_path):
            shutil.copy(backup_path, USER_PROFILE_PATH)
            break
        return True
    return False


def use_default_or_prompt_for_profile():
    """Attempt to use the default profile. If not valid, prompt the user to create a profile based on the template."""
    if os.path.exists(DEFAULT_PROFILE_PATH) and is_profile_valid(DEFAULT_PROFILE_PATH):
        shutil.copy(DEFAULT_PROFILE_PATH, USER_PROFILE_PATH)
    else:
        prompt_user_to_create_profile


def prompt_user_to_create_profile():
    """Print instructions for creating a profile from the template and loop until a valid profile is provided."""
    print(
        f"No valid default user profile found. Please create or update {USER_PROFILE_PATH}."
    )
    print(
        "Below is a profile template. Please use it to create 'user_profile.txt' and adapt it as needed:"
    )
    print(PROFILE_TEMPLATE_TEXT)

    while not is_profile_valid(USER_PROFILE_PATH):
        input("Press Enter after updating the profile to continue...")


if __name__ == "__main__":
    # instantiate ChromaDB
    persist_directory = "chromadb"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(name="knowledge_base")

    # instantiate chatbot
    the_chatbot = Chatbot(model=LLM_main, streaming=False)
    # chatbot_profile = Chatbot(model=LLM_profile)
    # chatbot_kb = Chatbot(model=LLM_kb)
    user_messages = list()
    all_messages = list()
    conversation_history = open_file(CONVERSATION_HISTORY_PATH)
    if conversation_history:
        conversation_history = json.loads(conversation_history)
        the_chatbot.conversation.set_history(conversation_history)

    while True:
        conversation_history = the_chatbot.conversation.history_log()
        # get user input
        while True:
            text = multi_line_input("\n\nUSER: ")
            if text.strip():
                break
        user_messages.append(text)
        all_messages.append("USER: %s" % text)
        save_file("chat_logs/chat_%s_user.txt" % time(), text)

        # update main scratchpad
        if len(all_messages) > 5:
            all_messages.pop(0)
        main_scratchpad = "\n\n".join(all_messages).strip()

        # search KB, update default system
        ensure_valid_user_profile()
        current_profile = open_file(USER_PROFILE_PATH)
        kb = "No KB articles yet"
        if collection.count() > 0:
            results = collection.query(query_texts=[main_scratchpad], n_results=1)
            kb = results["documents"][0][0]
        default_system = (
            open_file("prompts/system_default.txt")
            .replace("<<PROFILE>>", current_profile)
            .replace("<<KB>>", kb)
        )
        the_chatbot.add_prompt_to_conversation("system", default_system)

        # generate a response
        response = chatbot(the_chatbot, text)
        conversation_history = the_chatbot.conversation.history_log()
        save_file(CONVERSATION_HISTORY_PATH, json.dumps(conversation_history))
        all_messages.append("CHATBOT: %s" % response)
        print("\n\nCHATBOT: %s" % response)

        # update user scratchpad
        if len(user_messages) > 3:
            user_messages.pop(0)
        user_scratchpad = "\n".join(user_messages).strip()

        # update user profile
        print("\n\nUpdating user profile...")
        backup_current_profile()
        ensure_valid_user_profile()
        current_profile = open_file(USER_PROFILE_PATH)
        profile_length = len(current_profile.split(" "))
        profile_system_prompt = (
            open_file("prompts/system_update_user_profile.txt")
            .replace("<<UPD>>", current_profile)
            .replace("<<WORDS>>", str(profile_length))
        )
        the_chatbot.conversation.clear_history()
        the_chatbot.add_prompt_to_conversation("system", profile_system_prompt)
        updated_profile = chatbot(the_chatbot, text)
        if updated_profile.strip():
            save_file("user_profile.txt", updated_profile)
        else:
            print("Error: Updated profile is empty.")

        # update main scratchpad
        if len(all_messages) > 5:
            all_messages.pop(0)
        main_scratchpad = "\n\n".join(all_messages).strip()

        # Update the knowledge base
        print("\n\nUpdating KB...")
        if collection.count() == 0:
            # yay first KB!
            kb_system_prompt = open_file("prompts/system_instantiate_new_kb.txt")
            the_chatbot.conversation.clear_history()
            the_chatbot.add_prompt_to_conversation("system", kb_system_prompt)
            article = chatbot(the_chatbot, text)
            new_id = str(uuid4())
            collection.add(documents=[article], ids=[new_id])
            save_file(
                "db_logs/log_%s_add.txt" % time(),
                "Added document %s:\n%s" % (new_id, article),
            )
        else:
            results = collection.query(query_texts=[main_scratchpad], n_results=1)
            kb = results["documents"][0][0]
            kb_id = results["ids"][0][0]

            # Expand current KB
            kb_system_prompt = open_file(
                "prompts/system_update_existing_kb.txt"
            ).replace("<<KB>>", kb)
            the_chatbot.conversation.clear_history()
            the_chatbot.add_prompt_to_conversation("system", kb_system_prompt)
            article = chatbot(the_chatbot, text)
            collection.update(ids=[kb_id], documents=[article])
            save_file(
                "db_logs/log_%s_update.txt" % time(),
                "Updated document %s:\n%s" % (kb_id, article),
            )
            # TODO - save more info in DB logs, probably as YAML file (original article, new info, final article)

            # Split KB if too large
            kb_len = len(article.split(" "))
            if kb_len > 1000:
                kb_system_prompt = open_file("prompts/system_split_kb.txt")
                the_chatbot.conversation.clear_history()
                the_chatbot.add_prompt_to_conversation("system", kb_system_prompt)
                articles = chatbot(the_chatbot, text).split("ARTICLE 2:")
                a1 = articles[0].replace("ARTICLE 1:", "").strip()
                a2 = articles[1].strip()
                collection.update(ids=[kb_id], documents=[a1])
                new_id = str(uuid4())
                collection.add(documents=[a2], ids=[new_id])
                save_file(
                    "db_logs/log_%s_split.txt" % time(),
                    "Split document %s, added %s:\n%s\n\n%s" % (kb_id, new_id, a1, a2),
                )
