class ChatUI {
	constructor() {
		this.chatbox = document.getElementById("chatbox");
		this.userInput = document.getElementById("userInput");
		this.sendButton = document.getElementById("sendButton");
		this.modelSelect = document.getElementById('modelSelect');
		this.chatManager = null;
		this.setupEventListeners();
		this.userInput.focus();
	}

	setChatManager(chatManager) {
		this.chatManager = chatManager;
	}

	setupEventListeners() {
		this.userInput.addEventListener('keypress', e => this.handleKeypress(e));
		this.sendButton.addEventListener('click', () => this.handleSubmit());
		this.modelSelect.addEventListener('change', e => this.handleModelChange(e));
		this.userInput.addEventListener('input', () => this.resizeTextarea()); // Listen for input events to resize textarea
		window.addEventListener('resize', () => this.resizeTextarea()); // Update textarea height on window resize
	}

	handleKeypress(e) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			this.handleSubmit();
		}
	}

	handleSubmit() {
		let message = this.userInput.value.trim().replace(/\n/g, '<br>');
		if (message && this.chatManager) {
			this.chatManager.sendMessage(message);
			this.clearInput();
		}
	}

	handleModelChange(e) {
		if (this.chatManager) {
			this.chatManager.switchModel(e.target.value);
		}
	}

	renderMessage(sender, message, isNewMessage = true) {
		const content = UIHelpers.formatContent(message);
		const messageHtml = `<div class="${sender.toLowerCase()}-message">
			<p><strong>${sender}:</strong></p>${content}</div>`;
		this.chatbox.insertAdjacentHTML('beforeend', messageHtml);
		UIHelpers.highlightContent(this.chatbox.lastChild);
		this.scrollChatToBottom();
	}

	clearInput() {
		this.userInput.value = '';
		this.resizeTextarea();  // Reset the height after clearing
	}

	resizeTextarea() {
		this.userInput.style.height = 'auto'; // Reset the height
		let maxHeight = window.innerHeight / 3; // Calculate one third of the window height
		this.userInput.style.maxHeight = `${maxHeight}px`; // Set max height dynamically
		this.userInput.style.height = `${Math.min(this.userInput.scrollHeight, maxHeight)}px`; // Set the height to the lesser of scrollHeight or maxHeight
	}

	scrollChatToBottom() {
		this.chatbox.scrollTop = this.chatbox.scrollHeight;
	}
}

class UIHelpers {
	static formatContent(text) {
		const decodedText = this.decodeHtmlEntities(text);
		return `<div class="message-content"><p>${marked.parse(decodedText)}</p></div>`;
	}

	static decodeHtmlEntities(text) {
		const textarea = document.createElement('textarea');
		textarea.innerHTML = text;
		return textarea.value;
	}

static highlightContent(container) {
		// First, ensure that any code content is wrapped in <pre><code> for highlighting to work properly
		container.querySelectorAll('.message-content').forEach(content => {
			// This assumes that the markdown parsing correctly wraps code blocks in <code> tags
			content.querySelectorAll('code').forEach(codeBlock => {
				const pre = document.createElement('pre');
				codeBlock.parentNode.replaceChild(pre, codeBlock);
				pre.appendChild(codeBlock);
				hljs.highlightElement(codeBlock);
				this.setupCopyFunctionality(pre);
			});
		});
	}

	static setupCopyFunctionality(preElement) {
		const codeBlock = preElement.querySelector('code');
		if (!codeBlock) return;

		const toolbar = document.createElement('div');
		toolbar.className = 'code-header';
		const languageLabel = document.createElement('span');
		languageLabel.className = 'language-label';
		// Assuming codeBlock class contains language info, e.g., 'language-js'
		languageLabel.textContent = (codeBlock.className.match(/language-(\w+)/) || [,''])[1].toUpperCase();
		const copyButton = document.createElement('button');
		copyButton.className = 'copy-code-button';
		copyButton.textContent = 'Copy';
		copyButton.onclick = () => {
			navigator.clipboard.writeText(codeBlock.textContent).then(() => {
				copyButton.textContent = 'Copied!';
				setTimeout(() => copyButton.textContent = 'Copy', 3000);
			}).catch(err => console.error('Failed to copy text:', err));
		};
		toolbar.appendChild(languageLabel);
		toolbar.appendChild(copyButton);
		preElement.insertBefore(toolbar, preElement.firstChild);
	}
}

class MockChatManager {
	constructor(chatUI) {
		this.chatUI = chatUI;
	}

	sendMessage(message) {
		// console.log("Sending message:", message);
		// Display user message immediately
		this.chatUI.renderMessage("You", message, true);

		// Simulate receiving a response after a delay
		setTimeout(() => {
			this.chatUI.renderMessage("Chatbot", "This is a simulated response to: " + message, true);
		}, 1000);
	}

	switchModel(model) {
		console.log("Switching model to:", model);
	}
}

document.addEventListener('DOMContentLoaded', () => {
	const chatUI = new ChatUI();
	const chatManager = new MockChatManager(chatUI);
	chatUI.setChatManager(chatManager);
});
