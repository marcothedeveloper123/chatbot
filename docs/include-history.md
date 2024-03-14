# Include chat history

## requirements

**user story**
As a user
I want my chatbot to have access to our full conversation history
So I may re-start the chatbot and proceed from where we ended the conversation last time

**acceptance criteria**
given a third party module
when the module instantiates the chatbot
then the chatbot enables including chat history in the correct format

given a chatbot instantiation
when the third party module includes the chat history
then the chatbot checks that the chat history is valid

given a chatbot with uploaded chat history
when the user asks for recall from a past interaction inside of the context window
then the chatbot is able to access that interaction

To implement the requirements for managing and utilizing conversation history in the chatbot framework, we need to understand how the `Conversation` class operates within the context of these requirements. This documentation outlines the flow of conversation history, detailing how it is created, managed, and utilized by various components of the system.

## Overview

Analyzing your requirements for managing `conversation_history` in a chatbot framework, it's clear you aim for flexibility and robustness in how the conversation history is managed and utilized. Let's break down the requirements and their implications:

### Requirements Rephrased

1. **Conversation History on Instantiation**:
    - Allow the initialization of a chatbot with pre-existing conversation history. This means when a chatbot instance is created, it can be immediately equipped with a history that informs its context, enabling it to respond in a manner that's aware of previous interactions.

2. **Dynamic Conversation History Management**:
    - Implement a method, `.conversation.set_history()`, enabling the dynamic updating of the chatbot's conversation history. This feature should allow the chatbot to adapt to different contexts or user needs by loading different conversation histories as needed.

3. **History Validation**:
    - Ensure that any uploaded or set conversation history is valid. This entails checking the structure and content of the history to ensure it meets the expected format (e.g., correct keys and value types) and possibly the logical consistency of the conversation flow.

4. **Accessibility and Usage**:
    - Once the conversation history is set or uploaded, the chatbot must be able to access and effectively use this history for generating contextually relevant responses.

5. **Maintaining Logical Conversation Flow**:
    - If the most recent entry in the provided conversation history does not have the role `assistant`, remove entries from the end until an `assistant` role entry is the most recent. This ensures the chatbot is always responding to the last user prompt or system message, maintaining a logical flow of dialogue.

### Feasibility and Impact

- **Initialization with History**: Modifying the chatbot's constructor to accept a conversation history parameter is straightforward and requires ensuring that the history format matches what the `Conversation` class expects. This approach impacts how chatbot instances are created and initialized but doesn't fundamentally change the architecture.

- **Dynamic History Management**: Implementing a method to dynamically set conversation history is also feasible. The primary considerations here are validating the new history and resetting relevant internal state (e.g., token counts) to reflect the new history. This feature enhances the flexibility and reusability of chatbot instances.

- **History Validation**: Validating the conversation history is critical for ensuring the chatbot operates correctly and avoids errors during operation. This may introduce complexity, as validation needs to be thorough, checking not just the structure but also the integrity and logical flow of the conversation. It may require developing a detailed schema or set of rules against which the history is validated.

- **Accessibility and Usage Concerns**: Ensuring the chatbot can access and use the uploaded history involves integrating this history into the chatbot's response generation logic. This is inherently supported by the design but requires careful management of the internal state to ensure consistency with the newly set history.

- **Logical Conversation Flow Maintenance**: The requirement to ensure the most recent role is `assistant` before the chatbot takes over implies a manipulation of the conversation history that could remove user or system prompts if not carefully managed. This manipulation must be done thoughtfully to avoid losing important context or information from the conversation history.

### Under-Represented Issues

- **Error Handling and Feedback**: The requirements should consider how the system will handle invalid history inputs and provide feedback or error messages to the user or calling code. This includes how granular and informative the validation errors should be.
  
- **Historical Context Sensitivity**: Changing the conversation history on the fly can significantly alter the chatbot's context sensitivity. There's a need for mechanisms to ensure the chatbot's response generation is appropriately reset or adapted to the new history, especially regarding context understanding and token count management.

- **Performance Considerations**: Loading and validating large conversation histories dynamically could impact performance. Efficient data structures and algorithms are needed to minimize this impact, especially if the chatbot operates in real-time or near-real-time environments.

In summary, the proposed features for managing conversation history are feasible and would significantly enhance the flexibility and effectiveness of the chatbot framework. However, they introduce complexities around validation, state management, and ensuring logical conversation flow that must be carefully addressed to maintain the integrity and usability of the chatbot.