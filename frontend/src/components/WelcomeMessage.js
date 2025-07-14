// components/WelcomeMessage.js
import React from 'react';

const WelcomeMessage = ({ onSuggestionClick }) => {
  const quickSuggestions = [
    "Find milk promotions",
    "Compare bread prices across stores",
    "Healthy Italian dinner recipes",
    "Meals I can make with chicken and rice",
    "Vegetarian lunch ideas",
    "Cheapest pasta options"
  ];

  return (
    <div className="welcome-message">
      <div className="welcome-avatar">🛒</div>
      <h3>How can I help you today?</h3>
      <p>I can find promotions, compare prices across stores, or help you discover meals with grocery cost analysis.</p>
      
      <div className="quick-suggestions">
        <h4>Try asking:</h4>
        <div className="suggestions-grid">
          {quickSuggestions.map((suggestion, index) => (
            <button
              key={index}
              className="suggestion-btn"
              onClick={() => onSuggestionClick(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default WelcomeMessage;