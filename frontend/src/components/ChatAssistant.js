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
    inputRef.current?.focus();
  }, []);

  const quickQuestions = [
    "Find cheapest milk",
    "Today's promotions", 
    "Compare bread prices",
    "Budget meal for 2 people"
  ];

  // Simple markdown processor for basic formatting
  const processMarkdown = (text) => {
    if (!text) return '';
    
    // Convert markdown to HTML
    let html = text
      // Headers
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      // Bold text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Line breaks
      .replace(/\n/g, '<br>')
      // Lists (basic)
      .replace(/^- (.*$)/gim, '<li>$1</li>')
      .replace(/^(\d+)\. (.*$)/gim, '<li>$1. $2</li>')
      // Wrap consecutive <li> elements in <ul>
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
      // Clean up multiple <br> tags
      .replace(/(<br>\s*){3,}/g, '<br><br>');
    
    return html;
  };

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
          timestamp: new Date(),
          isMarkdown: true // Flag to indicate this needs markdown processing
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Error: ${response.error}`,
          error: true,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Connection error: ${error.message}`,
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

  const formatTime = (timestamp) => {
    return timestamp ? timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  };

  const renderMessageContent = (msg) => {
    if (msg.isMarkdown) {
      // Process markdown and render as HTML
      const htmlContent = processMarkdown(msg.content);
      return (
        <div 
          className="message-content markdown-content"
          dangerouslySetInnerHTML={{ __html: htmlContent }}
        />
      );
    } else {
      // Regular text content
      return (
        <div className="message-content">
          {msg.content}
        </div>
      );
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-avatar">🛒</div>
            <h3>Slovenian Grocery Assistant</h3>
            <p>Find the best prices across Slovenia's grocery stores</p>
            
            <div className="quick-questions">
              <h4>Try asking:</h4>
              <div className="quick-questions-grid">
                {quickQuestions.map((question, index) => (
                  <button
                    key={index}
                    className="quick-question-btn"
                    onClick={() => sendMessage(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>

            <div className="message-bubble">
              {renderMessageContent(msg)}
              {msg.timestamp && (
                <div className="message-time">
                  {formatTime(msg.timestamp)}
                </div>
              )}
            </div>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : '🛒'}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message assistant">
            <div className="message-bubble">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
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
            placeholder="Ask about grocery prices, deals, or products..."
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
              {loading ? '⏳' : '→'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatAssistant;