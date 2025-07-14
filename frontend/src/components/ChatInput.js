// components/ChatInput.js
import React from 'react';

const ChatInput = ({ 
  inputValue, 
  setInputValue, 
  onSendMessage, 
  loading, 
  inputRef 
}) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSendMessage();
    }
  };

  return (
    <div className="chat-input">
      <div className="input-container">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me anything about groceries, prices, or meals..."
          disabled={loading}
          maxLength={500}
        />
        <div className="input-actions">
          <span className="char-counter">{inputValue.length}/500</span>
          <button 
            className="send-button"
            onClick={onSendMessage}
            disabled={loading || !inputValue.trim()}
          >
            {loading ? '⏳' : '→'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;