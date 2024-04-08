/**
 * Chat class to manage chat interactions and UI updates.
 */
class Chat {
	/**
	 * Constructor initializes the chat application.
	 */
	constructor() {
		this.socket = io.connect(`http://${document.domain}:${location.port}`);
		// Constants for icon HTML
		this.USER_ICON_HTML = '<span class="material-symbols-outlined user-icon">&#xe7fd;</span>';
		this.CHATBOT_ICON_HTML = '<span class="material-symbols-outlined chatbot-icon">&#xe545;</span>';
		// Constants for class names
		this.USER_MESSAGE_CLASS = 'user-message';
		this.CHATBOT_RESPONSE_CLASS = 'chatbot-response';
		this.MESSAGE_CONTENT_CLASS = 'message-content';
		this.SENDER_NAME_CLASS = 'sender-name';

		// Populate the select box with all available LLM models
		this.modelSelect = document.getElementById('modelSelect');
		this.fetchAndPopulateModels();

		this.notyf = new Notyf();
		this.initChat();
	}

	/**
	 * Sets up the chat UI and binds event listeners.
	 */
	initChat() {
		// Initialize UI components.
		this.chatbox = document.getElementById("chatbox");
		this.userInput = document.getElementById("userInput");
		this.sendButton = document.getElementById("sendButton");

		// Bind UI events for chat interaction.
		this.bindEvents();
		// Restore previous chat state if available.
		this.restoreChatState();
		// Attach event listeners to existing Copy buttons
		this.attachListenersToCopyButtons();
		// Focus on the input field for immediate typing.
		this.userInput.focus();
	}

	/**
	 * Attaches event listeners for chat functionality.
	 */
	bindEvents() {
		// Handle message submission on 'Enter' press.
		this.userInput.addEventListener('keypress', (e) => {
			if (e.key === 'Enter' && this.userInput.value.trim()) {
				this.sendMessage(this.userInput.value.trim());
			}
		});

		// Handle message submission on send button click.
		this.sendButton.addEventListener('click', () => {
			if (this.userInput.value.trim()) {
				this.sendMessage(this.userInput.value.trim());
			}
		});

		// Save the current scroll position of the chatbox.
		this.chatbox.addEventListener('scroll', () => {
			localStorage.setItem('chatScrollPosition', this.chatbox.scrollTop);
		});

		// Error handling for socket connection
		this.socket.on('connect_error', () => {
			this.notifyUser("Connection error. Please try again.", 'error');
		});

		// Socket.io event listeners for receiving chatbot responses.
		this.socket.on('chatbot_response', (data) => {
			this.addMessage("Chatbot", data.message.replace(/\n/g, '<br>'), false);
		});

		this.socket.on('end_chatbot_response', () => {
			// Find the last chatbot response block in `#chatbox`
			const chatbotResponseElement = this.chatbox.querySelector(`.${this.CHATBOT_RESPONSE_CLASS}:last-of-type`);

			if (chatbotResponseElement) {
				// Get the message content element
				const messageContentElement = chatbotResponseElement.querySelector(`.${this.MESSAGE_CONTENT_CLASS}`);
				console.log(messageContentElement);

				// Parse the entire content of the last chatbot response as Markdown
				const parsedContent = marked.parse(messageContentElement.innerHTML.replace(/<br>/g, '\n'))
					.replace("&amp;", "&")
					.replace("&lt;", "<")
					.replace("&gt;", ">")
					.replace("&apos;", "'");

				// Replace the inner HTML of the message content element with the parsed content
				messageContentElement.innerHTML = parsedContent;

				hljs.highlightAll();

				// Set up toolbars for code blocks within the last chatbot response
				this.setupCodeBlockToolbars(chatbotResponseElement);
			}

			this.sendButton.disabled = false; // Re-enable the send button.
			this.saveChatState(); // Save the current chat state.
		});

		this.modelSelect.addEventListener('change', (event) => {
			const newModel = event.target.value;
			if (newModel) {
				this.switchModel(newModel);
			}
		});
	}

	/**
	 * Attach the "Copied!" functionality to all copy buttons.
	 */
	attachListenersToCopyButtons() {
		const copyButtons = this.chatbox.querySelectorAll('.copy-code-button');
		copyButtons.forEach(button => {
			// Directly passing the button to attachCopyButtonListener assumes
			// the method is adapted to accept a button element as a parameter.
			this.attachCopyButtonListener(button);
		});
	}

	attachCopyButtonListener(copyButton, codeBlock) {
		copyButton.addEventListener('click', () => {
			// After copying, clear the selection
			if (document.selection) {
				document.selection.empty();
			} else if (window.getSelection) {
				window.getSelection().removeAllRanges();
			}

			// Change button text to "✓ Copied!" and revert back after 3 seconds
			const originalText = copyButton.textContent;
			copyButton.textContent = '✓ Copied!';
			setTimeout(() => {
				copyButton.textContent = originalText; // Revert to original text
			}, 3000); // 3000ms = 3 seconds
		});
	}

	setupCodeBlockToolbars(chatbotResponseElement) {
		const codeBlocks = chatbotResponseElement.querySelectorAll('pre code');
		codeBlocks.forEach((block) => {
			const language = block.classList[0].replace('language-', ''); // Assuming `hljs` adds the language as the second class
			const toolbar = document.createElement('div');
			toolbar.className = 'code-header';

			const languageLabel = document.createElement('span');
			languageLabel.className = 'language-label';
			languageLabel.textContent = language ? language.toLowerCase() : 'CODE'; // Default label
			toolbar.appendChild(languageLabel);

			const copyButton = document.createElement('button');
			copyButton.className = 'copy-code-button';
			copyButton.textContent = 'Copy';
			toolbar.appendChild(copyButton);

			// Insert the toolbar before the code block
			block.parentNode.insertBefore(toolbar, block);

			// Attach copy functionality
			new ClipboardJS(copyButton, {
				target: () => block
			});

			// Attach event listener for clearing selection after copy
			this.attachCopyButtonListener(copyButton, block);
		});
	}

	/**
	 * Populate the select box with the names of available LLMs.
	 */
	fetchAndPopulateModels() {
		fetch('/models')
			.then(response => response.json())
			.then(data => {
				const { available_models: models, current_model: currentModel } = data;
				models.forEach(model => {
					const option = document.createElement('option');
					option.value = model;
					option.textContent = model;
					option.selected = model === currentModel;
					this.modelSelect.appendChild(option);
				});
			})
			.catch(error => {
				console.error('Error fetching models: ', errror);
				this.notifyUser('Error fetching models. Please refresh the page.', 'error');
			})
	}

	switchModel(newModel) {
		this.socket.emit('switch_model', { model: newModel });
		this.userInput.focus();
	}

	/**
	 * Sends a user message to the server and updates the UI.
	 * @param {string} message - Message content to send.
	 */
	sendMessage(message) {
		if (!this.socket.connected) {
			this.notifyUser("Not connected to the server. Please check your internet connection.", 'error');
			return;
		}

		this.socket.emit('send_message', { message }, (response) => {
			// Callback to handle acknowledgment or error from server
			if (response && response.error) {
				this.notifyUser(response.error, 'error');
			}
		});
		this.addMessage("You", message);
		this.addChatbotPlaceholder(); // Prepare for chatbot response
		this.userInput.value = ''; // Clear the input field.
		this.sendButton.disabled = true; // Disable the send button.
	}

	/**
	 * Adds a message to the chat UI.
	 * @param {string} sender - The sender of the message ("You" or "Chatbot").
	 * @param {string} message - The message content.
	 * @param {boolean} isNewMessage - Flag indicating if this is a new message or part of an ongoing response.
	 */
	addMessage(sender, message, isNewMessage = true) {
		const messageHtml = this.messageFactory(sender, message, isNewMessage);
		this.chatbox.insertAdjacentHTML('beforeend', messageHtml);
		this.scrollChatToBottom();
	}

	/**
	 * Factory method to create HTML string for messages.
	 * @param {string} sender - The sender of the message.
	 * @param {string} message - The message content.
	 * @param {boolean} isNewMessage - Flag for new message.
	 * @returns {string} The HTML string for the message.
	 */
	messageFactory(sender, message, isNewMessage) {
		const iconHtml = sender === "You" ? this.USER_ICON_HTML : this.CHATBOT_ICON_HTML;
		const senderLabel = `<strong class="${this.SENDER_NAME_CLASS}">${sender}:</strong>`;

		const messageContent = `<div class="${this.MESSAGE_CONTENT_CLASS}"><p>${message}</p></div>`;

		if (sender === "You" || isNewMessage) {
			return `<div class="user-message"><p>${iconHtml}${senderLabel}</p>${messageContent}</div>`;
		} else {
			const lastParagraph = this.chatbox.querySelector(`.${this.CHATBOT_RESPONSE_CLASS}:last-child .${this.MESSAGE_CONTENT_CLASS}`);
			if (lastParagraph) {
				// Append new message content correctly within a <div> container
				lastParagraph.innerHTML += message;
				return ''; // No need to return new HTML if appending to an existing message
			}
		}
		// For a new chatbot message
		return `<div class="${this.CHATBOT_RESPONSE_CLASS}"><p>${iconHtml}${senderLabel}</p><br>${messageContent}</div>`;
	}

	/**
	 * Display error or success messages using Notyf.
	 */
	notifyUser(message, type) {
		if (type === 'error') {
			this.notyf.error(message);
		} else {
			this.notyf.success(message);
		}
	}

	/**
	 * Adds a placeholder for the chatbot's upcoming message.
	 */
	addChatbotPlaceholder() {
		const placeholderHtml = `<div class="chatbot-response"><p>${this.CHATBOT_ICON_HTML} <strong>Chatbot:</strong></p><div class="message-content"></div></div>`;
		this.chatbox.insertAdjacentHTML('beforeend', placeholderHtml);
		this.scrollChatToBottom();
	}

	/**
	 * Scrolls the chatbox to the bottom to show the most recent message.
	 */
	scrollChatToBottom() {
		this.chatbox.scrollTop = this.chatbox.scrollHeight;
	}

	/**
	 * Saves the current chatbox HTML content to local storage for persistence.
	 */
	saveChatState() {
		const chatContent = this.chatbox.innerHTML;
		localStorage.setItem('chat', chatContent);
	}

	/**
	 * Restores chat state from local storage if available.
	 */
	restoreChatState() {
		const savedChat = localStorage.getItem('chat');
		if (savedChat) {
			this.chatbox.innerHTML = savedChat;
		}

		// Restore scroll position
		const savedScrollPosition = localStorage.getItem('chatScrollPosition');
		if (savedScrollPosition) {
			this.chatbox.scrollTop = parseInt(savedScrollPosition, 10);
		}
	}
}

document.addEventListener('DOMContentLoaded', () => new Chat());
