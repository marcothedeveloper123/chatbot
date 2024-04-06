class Chat {
	constructor() {
		this.socket = io.connect(`http://${document.domain}:${location.port}`);
		this.userIconHtml = '<span class="material-symbols-outlined user-icon">&#xe7fd;</span>';
		this.chatbotIconHtml = '<span class="material-symbols-outlined chatbot-icon">&#xe545;</span>';
		this.initChat();
	}

	initChat() {
		this.chatbox = document.getElementById("chatbox");
		this.userInput = document.getElementById("userInput");
		this.sendButton = document.getElementById("sendButton");
		this.bindEvents();
		this.restoreChatState();
		this.userInput.focus();
	}

	bindEvents() {
		this.userInput.addEventListener('keypress', (e) => {
			if (e.key === 'Enter' && this.userInput.value.trim()) {
				this.sendMessage(this.userInput.value.trim());
			}
		});

		this.sendButton.addEventListener('click', () => {
			if (this.userInput.value.trim()) {
				this.sendMessage(this.userInput.value.trim());
			}
		});

		this.chatbox.addEventListener('scroll', () => {
			localStorage.setItem('chatScrollPosition', chatbox.scrollTop);
		});

		this.socket.on('chatbot_response', (data) => {
			this.addMessage("Chatbot", data.message.replace(/\n/g, '<br>'), false);
		});

		this.socket.on('end_chatbot_response', () => {
			this.sendButton.disabled = false;
			this.saveChatState();
		});
	}

	sendMessage(message) {
		this.socket.emit('send_message', {message});
		this.addMessage("You", message);
		this.addChatbotPlaceholder(); // Prepare for chatbot response
		this.userInput.value = '';
		this.sendButton.disabled = true;
	}

	addMessage(sender, message, isNewMessage = true) {
		const senderIconHtml = sender === "You" ? this.userIconHtml : this.chatbotIconHtml;
		const senderLabel = `<strong class="sender-name">${sender}:</strong>`;
		let messageContent = `<span class="message-content">${message}</span>`;

		if (sender === "You") {
			const messageHtml = `<p class="user-message">${senderIconHtml}${senderLabel}<br>${messageContent}</p>`;
			chatbox.insertAdjacentHTML('beforeend', messageHtml);
		} else {
			let lastParagraph = chatbox.querySelector('.chatbot-response:last-child');
			if (!lastParagraph || isNewMessage) {
				const messageHtml = `<p class="chatbot-response">${senderIconHtml}${senderLabel}<br>${messageContent}</p>`;
				chatbox.insertAdjacentHTML('beforeend', messageHtml);
			} else {
				let messageContainer = lastParagraph.querySelector('.message-content');
				messageContainer.innerHTML += `${message}`;
			}
		}
		this.scrollChatToBottom();
	}

	addChatbotPlaceholder() {
		const placeholderHtml = `<p class="chatbot-response">${this.chatbotIconHtml} <strong>Chatbot:</strong><br><span class="message-content"></span></p>`;
		this.chatbox.insertAdjacentHTML('beforeend', placeholderHtml);
		this.scrollChatToBottom();
	}

	scrollChatToBottom() {
		this.chatbox.scrollTop = this.chatbox.scrollHeight;
	}

	saveChatState() {
		localStorage.setItem('chat', this.chatbox.innerHTML);
	}

	restoreChatState() {
		const savedChat = localStorage.getItem('chat');
		if (savedChat) this.chatbox.innerHTML = savedChat;
		const savedScrollPosition = localStorage.getItem('chatScrollPosition');
		if (savedScrollPosition) this.chatbox.scrollTop = parseInt(savedScrollPosition, 10);
	}
}

document.addEventListener('DOMContentLoaded', () => new Chat());
