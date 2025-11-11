// PhotonInfer Multi-Chat Web Application
class PhotonInferChat {
    constructor() {
        // Use relative URLs so it works with both HTTP and HTTPS
        this.apiUrl = '';
        this.isGenerating = false;

        // Chat sessions management
        this.chatSessions = new Map(); // id -> ChatSession
        this.currentChatId = null;
        this.nextChatId = 1;
        this.hasActiveSession = false; // Track if user has created a session

        // Clear all saved chat history on page load to ensure fresh sessions
        this.clearAllStorage();

        this.initElements();
        this.initEventListeners();

        // Don't create initial chat session or load history
        // User must click "New Chat" to start
        this.showWelcomeScreen();
    }

    initElements() {
        // Sidebar elements
        this.newChatBtn = document.getElementById('new-chat');
        this.chatList = document.getElementById('chat-list');

        // Stats elements
        this.tokensPerSec = document.getElementById('tokens-per-sec');
        this.totalTokens = document.getElementById('total-tokens');
        this.latency = document.getElementById('latency');

        // Chat elements
        this.chatTitle = document.getElementById('chat-title');
        this.chatMessages = document.getElementById('chat-messages');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-message');
        this.deleteBtn = document.getElementById('delete-chat');
    }

    initEventListeners() {
        // New chat button
        this.newChatBtn.addEventListener('click', () => this.createNewChat());

        // Delete chat button
        this.deleteBtn.addEventListener('click', () => this.deleteCurrentChat());

        // Chat interactions
        this.sendBtn.addEventListener('click', () => this.sendMessage());

        // Enter to send (Shift+Enter for new line)
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Focus input on load
        this.userInput.focus();
    }

    clearAllStorage() {
        try {
            localStorage.removeItem('photonInferChats');
            localStorage.removeItem('photonInferCurrentChat');
            console.log('[INFO] Cleared all saved chat history');
        } catch (e) {
            console.error('Failed to clear storage:', e);
        }
    }

    showWelcomeScreen() {
        this.hasActiveSession = false;
        this.currentChatId = null;
        this.chatTitle.textContent = 'Welcome to PhotonInfer';
        this.chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    <strong>Welcome to PhotonInfer!</strong><br><br>
                    To start a new conversation, please click the <strong>"+ New Chat"</strong> button in the sidebar.<br><br>
                    This ensures each session gets a fresh start with the Llama 3.2 1B model.
                </div>
            </div>
        `;
        this.userInput.disabled = true;
        this.sendBtn.disabled = true;
        this.deleteBtn.disabled = true;
    }

    createNewChat() {
        // Generate unique session ID with timestamp and random string
        const timestamp = Date.now();
        const randomStr = Math.random().toString(36).substring(2, 9);
        const chatId = `chat-${timestamp}-${randomStr}`;

        const chatSession = {
            id: chatId,
            title: 'New Conversation',
            messages: [],
            conversationHistory: [],
            createdAt: timestamp,
            updatedAt: timestamp
        };

        this.chatSessions.set(chatId, chatSession);
        this.hasActiveSession = true;
        this.userInput.disabled = false;
        this.sendBtn.disabled = false;
        this.deleteBtn.disabled = false;
        this.switchToChat(chatId);
        this.updateChatList();
        // Don't save to localStorage - sessions are ephemeral
    }

    switchToChat(chatId) {
        this.currentChatId = chatId;
        const session = this.chatSessions.get(chatId);

        if (!session) return;

        // Update title
        this.chatTitle.textContent = session.title;

        // Clear and rebuild messages
        this.chatMessages.innerHTML = '';

        if (session.messages.length === 0) {
            this.addSystemMessage('Welcome to PhotonInfer! The Llama 3.2 1B model is ready. Start chatting below.');
        } else {
            // Restore messages
            session.messages.forEach(msg => {
                this.appendMessage(msg.role, msg.content, false);
            });
        }

        // Update chat list active state
        this.updateChatList();
    }

    deleteCurrentChat() {
        if (this.chatSessions.size === 0) {
            return;
        }

        if (!confirm('Are you sure you want to delete this chat?')) {
            return;
        }

        // Clear session on server
        this.clearServerSession(this.currentChatId);

        this.chatSessions.delete(this.currentChatId);

        // If no more chats, show welcome screen
        if (this.chatSessions.size === 0) {
            this.showWelcomeScreen();
        } else {
            // Switch to another chat
            const firstChatId = this.chatSessions.keys().next().value;
            this.switchToChat(firstChatId);
        }

        this.updateChatList();
    }

    async clearServerSession(sessionId) {
        try {
            await fetch(`${this.apiUrl}/api/clear_session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });
        } catch (error) {
            console.error('Error clearing server session:', error);
        }
    }

    updateChatList() {
        this.chatList.innerHTML = '';

        // Sort chats by updated time (most recent first)
        const sortedChats = Array.from(this.chatSessions.values())
            .sort((a, b) => b.updatedAt - a.updatedAt);

        sortedChats.forEach(session => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            if (session.id === this.currentChatId) {
                chatItem.classList.add('active');
            }

            const preview = this.getChatPreview(session);

            chatItem.innerHTML = `
                <div class="chat-item-title">${this.escapeHtml(session.title)}</div>
                <div class="chat-item-preview">${this.escapeHtml(preview)}</div>
            `;

            chatItem.addEventListener('click', () => {
                this.switchToChat(session.id);
            });

            this.chatList.appendChild(chatItem);
        });
    }

    getChatPreview(session) {
        if (session.messages.length === 0) {
            return 'No messages yet';
        }

        // Get last user or assistant message
        for (let i = session.messages.length - 1; i >= 0; i--) {
            const msg = session.messages[i];
            if (msg.role === 'user' || msg.role === 'assistant') {
                return msg.content.substring(0, 40) + (msg.content.length > 40 ? '...' : '');
            }
        }

        return 'New conversation';
    }

    async sendMessage() {
        // Check if user has an active session
        if (!this.hasActiveSession || !this.currentChatId) {
            this.addSystemMessage('Please click "+ New Chat" to start a new conversation.');
            return;
        }

        const message = this.userInput.value.trim();

        if (!message || this.isGenerating) {
            return;
        }

        const session = this.chatSessions.get(this.currentChatId);
        if (!session) return;

        // Add user message to chat
        this.addUserMessage(message);
        this.userInput.value = '';

        session.conversationHistory.push({ role: 'user', content: message });
        session.messages.push({ role: 'user', content: message });
        session.updatedAt = Date.now();

        // Update chat title if this is the first message
        if (session.conversationHistory.length === 1) {
            session.title = message.substring(0, 30) + (message.length > 30 ? '...' : '');
            this.chatTitle.textContent = session.title;
        }

        // Show typing indicator
        const typingId = this.addTypingIndicator();
        this.isGenerating = true;
        this.sendBtn.disabled = true;

        const startTime = Date.now();
        let generatedTokens = 0;

        try {
            // Build request with conversation history
            const requestBody = {
                message: message,
                session_id: this.currentChatId,
                conversation_history: session.conversationHistory.slice(0, -1), // All messages except current
                max_tokens: 512,
                temperature: 0.7
            };

            const response = await fetch(`${this.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            this.removeTypingIndicator(typingId);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const endTime = Date.now();
            const elapsed = (endTime - startTime) / 1000;

            // Add assistant response
            this.addAssistantMessage(data.response);
            session.conversationHistory.push({ role: 'assistant', content: data.response });
            session.messages.push({ role: 'assistant', content: data.response });
            session.updatedAt = Date.now();

            // Update stats
            generatedTokens = data.tokens || data.response.split(/\s+/).length;
            this.updateStats(generatedTokens, elapsed);

            // Save and update UI
            this.updateChatList();

        } catch (error) {
            console.error('Error sending message:', error);
            this.removeTypingIndicator(typingId);
            this.addSystemMessage('Error: Could not get response from server. Please make sure the server is running.');
        } finally {
            this.isGenerating = false;
            this.sendBtn.disabled = false;
            this.userInput.focus();
        }
    }

    addUserMessage(content) {
        this.appendMessage('user', content, true);
    }

    addAssistantMessage(content) {
        this.appendMessage('assistant', content, true);
    }

    addSystemMessage(content) {
        this.appendMessage('system', content, false);
    }

    appendMessage(role, content, saveToSession) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        let label = '';
        if (role === 'user') label = 'You:';
        else if (role === 'assistant') label = 'Assistant:';
        else if (role === 'system') label = 'System:';

        messageDiv.innerHTML = `
            <div class="message-content">
                <strong>${label}</strong>
                ${this.escapeHtml(content)}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        // Save to session if needed
        if (saveToSession) {
            const session = this.chatSessions.get(this.currentChatId);
            if (session && role !== 'system') {
                session.messages.push({ role, content });
                session.updatedAt = Date.now();
            }
        }
    }

    addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = id;
        messageDiv.className = 'message assistant-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <strong>Assistant:</strong>
                <span class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </span>
            </div>
        `;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        return id;
    }

    removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    updateStats(tokens, elapsed) {
        const tokensPerSec = (tokens / elapsed).toFixed(1);
        const totalTokens = parseInt(this.totalTokens.textContent) + tokens;

        this.tokensPerSec.textContent = `${tokensPerSec} tok/s`;
        this.totalTokens.textContent = totalTokens;
        this.latency.textContent = `${elapsed.toFixed(2)}s`;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Note: Chat history is NOT persisted to localStorage
    // Each page load starts fresh to ensure unique sessions
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new PhotonInferChat();
    console.log('PhotonInfer Multi-Chat initialized');
});
