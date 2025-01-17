let sessionId = null;

// Initialize the application
async function init() {
    try {
        const response = await fetch('/api/session', {
            method: 'POST'
        });
        const data = await response.json();
        sessionId = data.session_id;
    } catch (error) {
        console.error('Failed to initialize session:', error);
    }
}

// Send message to the server
async function sendMessage(message) {
    if (!sessionId) return;

    try {
        // Add user message to chat
        addMessageToChat(message, 'user');

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });

        const data = await response.json();

        // Add bot response to chat
        addMessageToChat(data.response, 'bot');

        // Update preferences if any were extracted
        if (data.preferences && data.preferences.length > 0) {
            updatePreferences(data.preferences);
        }
    } catch (error) {
        console.error('Failed to send message:', error);
        addMessageToChat('Sorry, something went wrong. Please try again.', 'bot');
    }
}

// Get recommendations based on preferences
async function getRecommendations() {
    if (!sessionId) return;

    try {
        const response = await fetch('/api/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: "get recommendations",
                session_id: sessionId
            })
        });

        const data = await response.json();
        addMessageToChat(data.recommendations, 'bot');
    } catch (error) {
        console.error('Failed to get recommendations:', error);
        addMessageToChat('Sorry, failed to get recommendations. Please try again.', 'bot');
    }
}

// Add a message to the chat container
function addMessageToChat(message, type) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', type);
    messageElement.textContent = message;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Update preferences list
function updatePreferences(preferences) {
    const preferencesContainer = document.getElementById('preferences-list');
    preferencesContainer.innerHTML = ''; // Clear current preferences

    preferences.forEach(preference => {
        const preferenceElement = document.createElement('div');
        preferenceElement.classList.add('preference-item');
        preferenceElement.textContent = preference;
        preferencesContainer.appendChild(preferenceElement);
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    init();

    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const recommendationsButton = document.getElementById('get-recommendations');

    // Send message on button click
    sendButton.addEventListener('click', () => {
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
            userInput.value = '';
        }
    });

    // Send message on Enter key
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const message = userInput.value.trim();
            if (message) {
                sendMessage(message);
                userInput.value = '';
            }
        }
    });

    // Get recommendations on button click
    recommendationsButton.addEventListener('click', getRecommendations);
});