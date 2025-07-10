import React, { useState, useRef, useEffect } from 'react';
import ApiService from '../services/api';

const ChatAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Focus input on mount
    inputRef.current?.focus();
  }, []);

  const quickQuestions = [
    { text: "Find cheapest milk 🥛", query: "Find the cheapest milk in Ljubljana" },
    { text: "Today's promotions 🎯", query: "Show me today's best promotions and discounts" },
    { text: "Budget meal plan 🍽️", query: "Create a budget meal plan for 2 people with 50 EUR" },
    { text: "Compare bread prices 🍞", query: "Compare bread prices across different stores" }
  ];

  const sendMessage = async (message = inputValue) => {
    if (!message.trim()) return;

    const userMessage = { role: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await ApiService.sendChatMessage(message);
      
      if (response.success) {
        const botMessage = { 
          role: 'assistant', 
          content: response.data.response,
          function_used: response.data.function_used,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `❌ Oops! ${response.error}`,
          error: true,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `❌ Connection error: ${error.message}`,
        error: true,
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleQuickQuestion = (query) => {
    sendMessage(query);
  };

  const formatTime = (timestamp) => {
    return timestamp ? timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">

            
            <div className="welcome-features">
              <div className="feature-grid">
                <div className="feature-item">
                  <span className="feature-icon">🔍</span>
                  <h4>Product Search</h4>
                  <p>Find cheapest prices instantly</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">💰</span>
                  <h4>Budget Planning</h4>
                  <p>Create optimized shopping lists</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🎁</span>
                  <h4>Live Promotions</h4>
                  <p>Never miss a great deal</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🏪</span>
                  <h4>Store Comparison</h4>
                  <p>Compare prices across stores</p>
                </div>
              </div>
            </div>

            <div className="quick-questions">
              <h4>Try these quick questions:</h4>
              <div className="quick-questions-grid">
                {quickQuestions.map((question, index) => (
                  <button
                    key={index}
                    className="quick-question-btn"
                    onClick={() => handleQuickQuestion(question.query)}
                  >
                    {question.text}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-bubble">
              <div className="message-content">
                {msg.content}
                {msg.function_used && (
                  <div className="function-used">
                    <span className="function-icon">⚡</span>
                    Used: {msg.function_used}
                  </div>
                )}
              </div>
              {msg.timestamp && (
                <div className="message-time">
                  {formatTime(msg.timestamp)}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-bubble">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <div className="typing-text">AI is thinking...</div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input">
        <div className="input-container">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about grocery prices, promotions, or meal planning..."
            disabled={loading}
            maxLength={500}
          />
          <div className="input-actions">
            <span className="char-counter">{inputValue.length}/500</span>
            <button 
              className="send-button"
              onClick={() => sendMessage()}
              disabled={loading || !inputValue.trim()}
            >
              <span className="send-icon">
                {loading ? '⏳' : '🚀'}
              </span>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatAssistant;