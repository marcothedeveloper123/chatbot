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
			this.sendButton.disabled = false; // Re-enable the send button.
			this.saveChatState(); // Save the current chat state.
		});
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

		this.socket.emit('send_message', {message}, (response) => {
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
		const messageContent = `<span class="${this.MESSAGE_CONTENT_CLASS}">${message}</span>`;

		if (sender === "You" || isNewMessage) {
			return `<p class="${sender.toLowerCase()}-message">${iconHtml}${senderLabel}<br>${messageContent}</p>`;
		} else {
			const lastParagraph = this.chatbox.querySelector(`.${this.CHATBOT_RESPONSE_CLASS}:last-child .${this.MESSAGE_CONTENT_CLASS}`);
			if (lastParagraph) {
				lastParagraph.innerHTML += message;
				return ''; // No need to return new HTML if appending to an existing message
			}
		}
		// For a new chatbot message
		return `<p class="${this.CHATBOT_RESPONSE_CLASS}">${iconHtml}${senderLabel}<br>${messageContent}</p>`;
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
		const placeholderHtml = `<p class="chatbot-response">${this.CHATBOT_ICON_HTML} <strong>Chatbot:</strong><br><span class="message-content"></span></p>`;
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
		localStorage.setItem('chat', this.chatbox.innerHTML);
	}

	/**
	 * Restores chat state from local storage if available.
	 */
	restoreChatState() {
		const savedChat = localStorage.getItem('chat');
		if (savedChat) this.chatbox.innerHTML = savedChat;
		const savedScrollPosition = localStorage.getItem('chatScrollPosition');
		if (savedScrollPosition) this.chatbox.scrollTop = parseInt(savedScrollPosition, 10);
	}
}

document.addEventListener('DOMContentLoaded', () => new Chat());
