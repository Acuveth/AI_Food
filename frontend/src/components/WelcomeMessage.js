// components/WelcomeMessage.js - Slovenian Language Support
import React from 'react';

const WelcomeMessage = ({ onSuggestionClick }) => {
  const quickSuggestions = [
    // Mix of Slovenian and English suggestions
    "Najdi akcije za mleko",
    "Primerjaj cene kruha v trgovinah",
    "Zdravi italijanski recepti za večerjo",
    "Kaj lahko skuham s piščancem in rižem",
    "Vegetarijski recepti za kosilo",
    "Kje najceneje kupim testenine",
    "Find cheese promotions",
    "Veganski recepti za zajtrk",
    "Akcije za sadje in zelenjavo",
    "Kaj pripravi iz krompirja",
    "Cheap organic products",
    "Recepti brez glutena"
  ];

  return (
    <div className="welcome-message">
      <div className="quick-suggestions">
        <h4>Poskusite vprašati:</h4>
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