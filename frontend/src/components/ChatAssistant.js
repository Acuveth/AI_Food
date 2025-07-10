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
    { text: "Find cheapest milk 🥛", query: "Najdi najcenejše mleko" },
    { text: "Today's promotions 🎯", query: "Pokaži današnje promocije in popuste" },
    { text: "Budget meal plan 🍽️", query: "Ustvari načrt obrokov za 2 osebi z 50 EUR" },
    { text: "Compare bread prices 🍞", query: "Primerjaj cene kruha v različnih trgovinah" },
    { text: "Healthy breakfast ideas 🌅", query: "Predlagaj zdrav zajtrk z visokimi zdravstvenimi ocenami" },
    { text: "Vegan products 🌱", query: "Najdi veganske izdelke" }
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
          function_result: response.data.function_result,
          semantic_validation: response.data.semantic_validation,
          validation_applied: response.validation_applied,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `❌ Ups! ${response.error}`,
          error: true,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `❌ Napaka povezave: ${error.message}`,
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

  const renderValidationInfo = (msg) => {
    if (!msg.validation_applied || !msg.function_result) return null;

    const result = msg.function_result;
    let validationContent = null;

    // Handle different types of function results
    if (result.success === false && result.suggestions) {
      // Search with suggestions
      validationContent = (
        <div className="validation-info suggestions">
          <div className="validation-header">
            <span className="validation-icon">🔍</span>
            <span>Semantična validacija</span>
          </div>
          <div className="validation-content">
            <p>Preprečil sem napačne zadetke za "{result.search_term}"</p>
            {result.raw_results_count > 0 && (
              <p>Našel {result.raw_results_count} izdelkov, toda nobeden ni ustrezal vašemu iskanju.</p>
            )}
            {result.suggestions && result.suggestions.length > 0 && (
              <div className="search-suggestions">
                <p><strong>Poskusite z:</strong></p>
                <div className="suggestion-buttons">
                  {result.suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      className="suggestion-btn"
                      onClick={() => sendMessage(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    } else if (result.success === true && result.validation_applied) {
      // Successful search with validation
      validationContent = (
        <div className="validation-info success">
          <div className="validation-header">
            <span className="validation-icon">✅</span>
            <span>Validacija uspešna</span>
          </div>
          <div className="validation-content">
            <p>Vrnil sem samo ustrezne izdelke za "{result.search_term}"</p>
            <p>Izključil sem napačne zadetke (npr. čokolada pri iskanju mleka)</p>
          </div>
        </div>
      );
    } else if (result.shopping_list_result?.validation_applied) {
      // Shopping list with validation
      const issues = result.shopping_list_result.validation_issues;
      if (issues && issues.length > 0) {
        validationContent = (
          <div className="validation-info warnings">
            <div className="validation-header">
              <span className="validation-icon">⚠️</span>
              <span>Validacijska opozorila</span>
            </div>
            <div className="validation-content">
              <ul>
                {issues.map((issue, index) => (
                  <li key={index}>{issue}</li>
                ))}
              </ul>
            </div>
          </div>
        );
      }
    }

    return validationContent;
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-avatar">🤖</div>
            <h3>Pozdravljeni v izboljšanem slovenskem grocery asistentintelligence!</h3>
            <p>Zdaj z <strong>semantično validacijo</strong> - ne več napačnih zadetkov!</p>
            
            <div className="validation-highlight">
              <h4>🔍 Nove možnosti validacije:</h4>
              <div className="validation-features">
                <div className="validation-feature">
                  <span className="feature-icon">✅</span>
                  <div>
                    <strong>Natančni zadetki</strong>
                    <p>Ko iščete "mleko", ne boste več dobili "MLEČNA REZINA MILKA"</p>
                  </div>
                </div>
                <div className="validation-feature">
                  <span className="feature-icon">💡</span>
                  <div>
                    <strong>Pametni predlogi</strong>
                    <p>Predlagam alternativne iskalne termine, če ni zadetkov</p>
                  </div>
                </div>
                <div className="validation-feature">
                  <span className="feature-icon">🎯</span>
                  <div>
                    <strong>Prepoznavanje namena</strong>
                    <p>Razumem, kaj dejansko iščete, ne samo besede</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="welcome-features">
              <div className="feature-grid">
                <div className="feature-item">
                  <span className="feature-icon">🔍</span>
                  <h4>Validiran iskanje</h4>
                  <p>Samo ustrezni izdelki</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">💰</span>
                  <h4>Pametno načrtovanje</h4>
                  <p>Optimizirani seznami za nakupovanje</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🎁</span>
                  <h4>Žive promocije</h4>
                  <p>Nikoli ne zamudite odličnih ponudb</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🏪</span>
                  <h4>Primerjava trgovin</h4>
                  <p>Cene v vseh glavnih trgovinah</p>
                </div>
              </div>
            </div>

            <div className="quick-questions">
              <h4>Preizkusite ta vprašanja:</h4>
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
                
                {/* Enhanced function usage display */}
                {msg.function_used && (
                  <div className="function-used">
                    <span className="function-icon">⚡</span>
                    Funkcija: {msg.function_used}
                    {msg.semantic_validation && (
                      <span className="validation-badge">
                        <span className="validation-icon">🔍</span>
                        Validacija
                      </span>
                    )}
                  </div>
                )}
                
                {/* Validation information */}
                {renderValidationInfo(msg)}
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
                <div className="typing-text">AI razmišlja in validira rezultate...</div>
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
            placeholder="Vprašajte o cenah, promocijah ali načrtovanju obrokov..."
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
              Pošlji
            </button>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .validation-highlight {
          background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1));
          border: 1px solid rgba(16, 185, 129, 0.2);
          border-radius: 16px;
          padding: 25px;
          margin: 30px 0;
          backdrop-filter: blur(10px);
        }
        
        .validation-highlight h4 {
          color: var(--text-primary);
          margin-bottom: 20px;
          font-size: 1.2em;
          text-align: center;
        }
        
        .validation-features {
          display: grid;
          gap: 15px;
        }
        
        .validation-feature {
          display: flex;
          align-items: flex-start;
          gap: 15px;
          padding: 15px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .validation-feature .feature-icon {
          font-size: 1.5em;
          flex-shrink: 0;
        }
        
        .validation-feature strong {
          color: var(--text-primary);
          font-size: 1.1em;
          display: block;
          margin-bottom: 5px;
        }
        
        .validation-feature p {
          color: var(--text-secondary);
          font-size: 0.9em;
          line-height: 1.4;
          margin: 0;
        }
        
        .validation-info {
          margin-top: 15px;
          padding: 15px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.2);
          backdrop-filter: blur(10px);
        }
        
        .validation-info.success {
          background: rgba(16, 185, 129, 0.1);
          border-color: rgba(16, 185, 129, 0.3);
        }
        
        .validation-info.suggestions {
          background: rgba(59, 130, 246, 0.1);
          border-color: rgba(59, 130, 246, 0.3);
        }
        
        .validation-info.warnings {
          background: rgba(245, 158, 11, 0.1);
          border-color: rgba(245, 158, 11, 0.3);
        }
        
        .validation-header {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 600;
          color: var(--text-primary);
          margin-bottom: 10px;
          font-size: 0.9em;
        }
        
        .validation-icon {
          font-size: 1.1em;
        }
        
        .validation-content {
          color: var(--text-secondary);
          font-size: 0.9em;
          line-height: 1.4;
        }
        
        .validation-content p {
          margin: 5px 0;
        }
        
        .validation-content ul {
          margin: 10px 0;
          padding-left: 20px;
        }
        
        .validation-content li {
          margin: 5px 0;
        }
        
        .search-suggestions {
          margin-top: 15px;
          padding-top: 15px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .suggestion-buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }
        
        .suggestion-btn {
          background: rgba(255, 255, 255, 0.1);
          color: var(--text-primary);
          border: 1px solid rgba(255, 255, 255, 0.2);
          padding: 6px 12px;
          border-radius: 15px;
          cursor: pointer;
          font-size: 0.85em;
          transition: all 0.3s ease;
          backdrop-filter: blur(5px);
        }
        
        .suggestion-btn:hover {
          background: rgba(59, 130, 246, 0.2);
          border-color: rgba(59, 130, 246, 0.4);
          transform: translateY(-2px);
        }
        
        .validation-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          margin-left: 10px;
          padding: 2px 8px;
          background: rgba(16, 185, 129, 0.2);
          border-radius: 10px;
          font-size: 0.75em;
          color: rgba(16, 185, 129, 0.9);
        }
        
        .validation-badge .validation-icon {
          font-size: 0.9em;
        }
        
        .function-used {
          display: flex;
          align-items: center;
          flex-wrap: wrap;
          gap: 5px;
        }
        
        @media (max-width: 768px) {
          .validation-features {
            grid-template-columns: 1fr;
          }
          
          .validation-feature {
            flex-direction: column;
            align-items: flex-start;
          }
          
          .validation-feature .feature-icon {
            margin-bottom: 5px;
          }
          
          .suggestion-buttons {
            flex-direction: column;
            align-items: stretch;
          }
          
          .suggestion-btn {
            text-align: center;
          }
        }
      `}</style>
    </div>
  );
};

export default ChatAssistant;