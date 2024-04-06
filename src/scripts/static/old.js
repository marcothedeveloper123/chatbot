document.addEventListener('DOMContentLoaded', function () {
	// Define icons HTML
	const userIconHtml = '<span class="material-symbols-outlined user-icon">&#xe7fd;</span>';
	const chatbotIconHtml = '<span class="material-symbols-outlined chatbot-icon">&#xe545;</span>';

	// Chat module for handling UI and chat interactions
	const chatModule = (function () {
function addMessage(sender, message, isNewMessage = false) {
		const senderIconHtml = sender === "You" ? userIconHtml : chatbotIconHtml;
		const senderLabel = `<strong class="sender-name">${sender}:</strong>`;
		let messageContent = `<span class="message-content">${message}</span>`;

		if (sender === "You") {
			const messageHtml = `<p class="user-message">${senderIconHtml}${senderLabel}<br>${messageContent}</p>`;
			chatbox.insertAdjacentHTML('beforeend', messageHtml);
		} else {
			let lastParagraph = chatbox.querySelector('.chatbot-response:last-child');
			if (!lastParagraph || isNewMessage) {
				// This condition is adjusted to handle the first token of a new chatbot message correctly.
				const messageHtml = `<p class="chatbot-response">${senderIconHtml}${senderLabel}<br>${messageContent}</p>`;
				chatbox.insertAdjacentHTML('beforeend', messageHtml);
			} else {
				// Existing logic for appending to last chatbot message goes here, ensuring it's placed below the "Chatbot:" header.
				let messageContainer = lastParagraph.querySelector('.message-content');
				messageContainer.innerHTML += `${message}`; // This now appends the message correctly under the header.
			}
		}
		scrollChatToBottom();
	}

		function scrollChatToBottom() {
			chatbox.scrollTop = chatbox.scrollHeight;
		}

		function saveChatState() {
			localStorage.setItem('chat', chatbox.innerHTML);
		}

		function restoreChatState() {
			const savedChat = localStorage.getItem('chat');
			if (savedChat) chatbox.innerHTML = savedChat;

			const savedScrollPosition = localStorage.getItem('chatScrollPosition');
			if (savedScrollPosition) chatbox.scrollTop = parseInt(savedScrollPosition, 10);
		}

		function addPlaceholder() {
			// const placeholderHtml = `<p class="chatbot-response">${chatbotIconHtml} <strong>Chatbot:</strong> <span class="message-content"></span><br></p>`;
			const placeholderHtml = `<p class="chatbot-response">${chatbotIconHtml} <strong>Chatbot:</strong><br><span class="message-content"></span></p>`;

			chatbox.insertAdjacentHTML('beforeend', placeholderHtml);
			scrollChatToBottom();
		}

		return {
			addMessage,
			addPlaceholder,
			restoreState: restoreChatState,
			saveState: saveChatState
		};
	})();

	// Socket.io communication
	const socket = io.connect(`http://${document.domain}:${location.port}`);
	const chatbox = document.getElementById("chatbox");
	const userInput = document.getElementById("userInput");
	const sendButton = document.getElementById("sendButton");

	// Event listeners
	userInput.addEventListener('keypress', function (e) {
		if (e.key === 'Enter' && userInput.value.trim()) {
			sendMessage(userInput.value.trim());
		}
	});

	sendButton.addEventListener('click', function () {
		if (userInput.value.trim()) {
			sendMessage(userInput.value.trim());
		}
	});

	chatbox.addEventListener('scroll', function () {
		localStorage.setItem('chatScrollPosition', chatbox.scrollTop);
	});

	// Function to send message
	function sendMessage(message) {
		socket.emit('send_message', { message });
		chatModule.addMessage("You", message);
		chatModule.addPlaceholder();
		userInput.value = '';
		sendButton.disabled = true; // Disable until chatbot responds
	}

	// Handling chatbot response
	socket.on('chatbot_response', function (data) {
		const message = data.message.replace(/\n/g, '<br>'); // Format for HTML
		chatModule.addMessage("Chatbot", message);
	});

	socket.on('end_chatbot_response', function () {
		sendButton.disabled = false; // Re-enable on chatbot's response end
		chatModule.saveState();
	});

	chatModule.restoreState(); // Restore chat history on page load
	userInput.focus();
});
