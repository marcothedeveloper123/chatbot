document.addEventListener('DOMContentLoaded', (event) => {
  const sendMessage = () => {
	const messageInput = document.getElementById('messageInput');
	const message = messageInput.value.trim();
	if (message) {
	  appendMessage('You', message);
	  messageInput.value = '';
	}
  };

  // Function to append messages to the conversation history
  const appendMessage = (sender, message) => {
	const conversationHistory = document.getElementById('conversationHistory');
	const messageDiv = document.createElement('div');
	messageDiv.classList.add('message', sender.toLowerCase());
	messageDiv.textContent = `${sender}: ${message}`;
	// Append message at the end of the conversationHistory
	conversationHistory.appendChild(messageDiv);
	// Scroll to the bottom of the conversation history
	conversationHistory.scrollTop = conversationHistory.scrollHeight;
  };

  document.getElementById('sendButton').addEventListener('click', sendMessage);

  // Optionally, listen for Enter to send message
  document.getElementById('messageInput').addEventListener('keypress', (e) => {
	if (e.key === 'Enter') {
	  e.preventDefault(); // Prevent the default action to avoid submitting the form
	  sendMessage();
	}
  });
});
