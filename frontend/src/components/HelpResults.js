// components/HelpResults.js - Slovenian Language Support
import React from 'react';

// General Help Results Component
const GeneralHelpResults = ({ data, onSuggestionClick }) => {
  const slovenianSuggestions = [
    "Najdi akcije za mleko",
    "Primerjaj cene kruha v trgovinah", 
    "Vegetarijski recepti za kosilo",
    "Kje najceneje kupim testenine",
    "Zdravi zajtrki za družino",
    "Akcije za sadje in zelenjavo",
    "Recepti brez glutena",
    "Kaj pripravi iz krompirja"
  ];

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>💡 Kako vam lahko pomagam?</h3>
        <p>
          {data.response || 
          "Pomagam vam najti akcije, primerjati cene ali iskati recepte. Poskusite vprašati nekaj kot 'najdi akcije za mleko', 'primerjaj cene kruha' ali 'italijanski recepti za večerjo'."}
        </p>
      </div>
      
      <div className="help-suggestions">
        <h4>Poskusite te primere:</h4>
        <div className="suggestions-grid">
          {(data.suggestions || slovenianSuggestions).map((suggestion, index) => (
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
      
      <div className="features-info">
        <h4>Kaj znam:</h4>
        <div className="features-list">
          <div className="feature-item">
            <span className="feature-icon">🏷️</span>
            <span>Iskanje akcij in popustov</span>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🔍</span>
            <span>Primerjanje cen med trgovinami</span>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🍽️</span>
            <span>Predlogi receptov z analizo stroškov</span>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🥗</span>
            <span>Iskanje jedi glede na sestavine</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Clarification Results Component
const ClarificationResults = ({ data, onSuggestionClick }) => {
  const defaultQuestions = [
    "Kateri specifični izdelek iščete?",
    "Želite najti akcije, primerjati cene ali dobiti predloge za recepte?",
    "Imate preference glede trgovine (DM, Lidl, Mercator, SPAR, Tuš)?",
    "Iščete kaj posebnega za določen obrok (zajtrk, kosilo, večerja)?"
  ];

  const slovenianSuggestions = [
    "Akcije za mleko v DM",
    "Primerjaj cene kruha",
    "Vegetarijski recepti",
    "Najcenejše testenine",
    "Zdravi zajtrki",
    "Recepti s piščancem",
    "Popusti za sadje",
    "Jedi brez glutena"
  ];

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🤔 Potrebujem več informacij</h3>
        <p>Ali lahko pojasnite, kaj iščete?</p>
      </div>
      
      <div className="clarification-questions">
        <h4>Nekaj vprašanj za pomoč:</h4>
        <div className="questions-list">
          {(data.clarification_questions || defaultQuestions).map((question, index) => (
            <div key={index} className="question-item">
              <span className="question-icon">❓</span>
              <span>{question}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="help-suggestions">
        <h4>Ali poskusite te primere:</h4>
        <div className="suggestions-grid">
          {(data.suggestions || slovenianSuggestions).map((suggestion, index) => (
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
      
      <div className="help-tips">
        <h4>💡 Nasveti za boljše iskanje:</h4>
        <ul>
          <li>Uporabite slovenska imena izdelkov (npr. "mleko" namesto "milk")</li>
          <li>Omenite lahko trgovino (npr. "DM", "Lidl", "Mercator")</li>
          <li>Povejte, če iščete akcije: "najdi akcije za..."</li>
          <li>Za recepte dodajte vrsto jedi: "recepti za večerjo"</li>
        </ul>
      </div>
    </div>
  );
};

export default GeneralHelpResults;
export { ClarificationResults };