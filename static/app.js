let sessionId = null;

async function init() {
    try {
        const response = await fetch('/api/session', {
            method: 'POST'
        });
        const data = await response.json();
        sessionId = data.session_id;

        // Add initial message
        addMessageToChat('Hi! Tell me about your cocktail preferences.', 'bot');
    } catch (error) {
        console.error('Failed to initialize session:', error);
    }
}

async function sendMessage(message) {
    if (!sessionId) return;

    try {
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

        // Update preferences panel with accumulated preferences
        if (data.preferences) {
            updatePreferences(data.preferences);
        }
    } catch (error) {
        console.error('Failed to send message:', error);
        addMessageToChat('Sorry, something went wrong. Please try again.', 'bot');
    }
}

function updatePreferences(preferences) {
    const preferencesContainer = document.getElementById('preferences-list');
    preferencesContainer.innerHTML = ''; // Clear current display

    // Helper function to add a category of preferences
    const addPreferenceCategory = (items, emoji, title) => {
        if (items && items.length > 0) {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'preference-category';
            categoryDiv.innerHTML = `
                <div class="preference-category-title">${title}</div>
                ${items.map(item => `
                    <div class="preference-item">
                        ${emoji} ${item}
                    </div>
                `).join('')}
            `;
            preferencesContainer.appendChild(categoryDiv);
        }
    };

    // Add each category with its own section
    addPreferenceCategory(preferences.liked_ingredients, '🌿', 'Favorite Ingredients');
    addPreferenceCategory(preferences.liked_cocktails, '🍸', 'Favorite Cocktails');
    addPreferenceCategory(preferences.liked_characteristics, '✨', 'Preferred Characteristics');
}

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

function resetSession() {
    const messagesContainer = document.getElementById('chat-messages');
    const preferencesContainer = document.getElementById('preferences-list');

    messagesContainer.innerHTML = '';
    preferencesContainer.innerHTML = '';

    init();
}

function addMessageToChat(message, type) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', type);
    messageElement.textContent = message;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

document.addEventListener('DOMContentLoaded', () => {
    init();

    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const recommendationsButton = document.getElementById('get-recommendations');
    const resetButton = document.getElementById('reset-session');

    sendButton.addEventListener('click', () => {
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
            userInput.value = '';
        }
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const message = userInput.value.trim();
            if (message) {
                sendMessage(message);
                userInput.value = '';
            }
        }
    });

    recommendationsButton.addEventListener('click', getRecommendations);
    resetButton.addEventListener('click', resetSession);
});