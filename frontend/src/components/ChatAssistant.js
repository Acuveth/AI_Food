import React, { useState } from 'react';
import ApiService from '../services/api';

const ChatAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await ApiService.sendChatMessage(inputValue);
      
      if (response.success) {
        const botMessage = { 
          role: 'assistant', 
          content: response.data.response,
          function_used: response.data.function_used 
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Error: ${response.error}` 
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${error.message}` 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h3>Welcome to Slovenian Grocery Intelligence!</h3>
            <p>Ask me about:</p>
            <ul>
              <li>Finding the cheapest products</li>
              <li>Comparing prices across stores</li>
              <li>Current promotions and discounts</li>
            </ul>
          </div>
        )}
        
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">
              {msg.content}
              {msg.function_used && (
                <div className="function-used">
                  Used: {msg.function_used}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about grocery prices..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatAssistant;