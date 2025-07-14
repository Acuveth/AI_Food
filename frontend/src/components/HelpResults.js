// components/HelpResults.js
import React from 'react';

// General Help Results Component
const GeneralHelpResults = ({ data, onSuggestionClick }) => {
  return (
    <div className="results-container">
      <div className="message-content">
        <h3>💡 How Can I Help?</h3>
        <p>{data.response}</p>
      </div>
      
      <div className="help-suggestions">
        <h4>Try these examples:</h4>
        <div className="suggestions-grid">
          {data.suggestions.map((suggestion, index) => (
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

// Clarification Results Component
const ClarificationResults = ({ data, onSuggestionClick }) => {
  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🤔 I need more information</h3>
        <p>Could you please clarify what you're looking for?</p>
      </div>
      
      <div className="clarification-questions">
        <h4>Some questions to help:</h4>
        <div className="questions-list">
          {data.clarification_questions?.map((question, index) => (
            <div key={index} className="question-item">
              {question}
            </div>
          ))}
        </div>
      </div>
      
      <div className="help-suggestions">
        <h4>Or try these examples:</h4>
        <div className="suggestions-grid">
          {data.suggestions?.map((suggestion, index) => (
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

export default GeneralHelpResults;
export { ClarificationResults };