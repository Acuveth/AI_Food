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
    { text: "Product insights 📊", query: "Pokaži vpogled v cene in kvaliteto mleka" }
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
          dynamic_validation: response.data.dynamic_validation,
          validation_applied: response.validation_applied,
          validation_details: response.validation_details,
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

  const renderDynamicValidationInfo = (msg) => {
    if (!msg.validation_applied && !msg.dynamic_validation) return null;

    const details = msg.validation_details;
    const result = msg.function_result;

    // Handle different types of dynamic validation results
    if (result?.success === false && result?.suggestions) {
      // No valid products found with suggestions
      return (
        <div className="validation-info dynamic-suggestions">
          <div className="validation-header">
            <span className="validation-icon">🧠</span>
            <span>Dinamična LLM validacija</span>
          </div>
          <div className="validation-content">
            <p>Analiziral sem bazo podatkov za "{result.search_term}" in ni našel ustreznih izdelkov.</p>
            {result.validation_reasoning && (
              <div className="validation-reasoning">
                <strong>Razlog:</strong> {result.validation_reasoning}
              </div>
            )}
            {result.raw_results_count > 0 && (
              <p>Našel sem {result.raw_results_count} izdelkov v bazi, a noben ni ustrezal vaši zahtevi.</p>
            )}
            {result.suggestions && result.suggestions.length > 0 && (
              <div className="smart-suggestions">
                <p><strong>Pametni predlogi na osnovi analize baze:</strong></p>
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
    } else if (result?.success === true && details) {
      // Successful search with dynamic validation details
      return (
        <div className="validation-info dynamic-success">
          <div className="validation-header">
            <span className="validation-icon">🎯</span>
            <span>Dinamična validacija uspešna</span>
          </div>
          <div className="validation-content">
            <p>Uporabil sem pametno LLM analizo za "{result.search_term}"</p>
            
            {details.reasoning && (
              <div className="validation-reasoning">
                <strong>Analiza:</strong> {details.reasoning}
              </div>
            )}
            
            <div className="validation-stats">
              {details.confidence && (
                <div className="confidence-score">
                  <span className="stat-label">Zaupanje validacije:</span>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ 
                        width: `${(details.confidence * 100)}%`,
                        backgroundColor: details.confidence >= 0.8 ? '#10b981' : 
                                       details.confidence >= 0.6 ? '#f59e0b' : '#ef4444'
                      }}
                    ></div>
                  </div>
                  <span className="confidence-value">{(details.confidence * 100).toFixed(0)}%</span>
                </div>
              )}
              
              {details.invalid_products_filtered > 0 && (
                <p>🚫 Izločil sem {details.invalid_products_filtered} nepomembnih izdelkov</p>
              )}
            </div>
          </div>
        </div>
      );
    } else if (result?.insights_result) {
      // Product insights with LLM analysis
      const insights = result.insights_result.insights;
      return (
        <div className="validation-info insights">
          <div className="validation-header">
            <span className="validation-icon">📊</span>
            <span>LLM Analiza Izdelka</span>
          </div>
          <div className="validation-content">
            {insights?.summary && (
              <div className="insights-summary">
                <strong>Povzetek:</strong> {insights.summary}
              </div>
            )}
            
            {insights?.price_analysis && (
              <div className="price-insights">
                <h5>💰 Cenovna analiza:</h5>
                <ul>
                  <li>Najcenejša cena: €{insights.price_analysis.cheapest_price?.toFixed(2)}</li>
                  <li>Najdražja cena: €{insights.price_analysis.most_expensive_price?.toFixed(2)}</li>
                  <li>Povprečna cena: €{insights.price_analysis.average_price?.toFixed(2)}</li>
                  {insights.price_analysis.best_value_store && (
                    <li>Najboljša vrednost: {insights.price_analysis.best_value_store}</li>
                  )}
                </ul>
              </div>
            )}
            
            {insights?.recommendations && insights.recommendations.length > 0 && (
              <div className="ai-recommendations">
                <h5>💡 AI priporočila:</h5>
                <ul>
                  {insights.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      );
    }

    // Generic dynamic validation badge
    if (msg.dynamic_validation) {
      return (
        <div className="validation-info dynamic-generic">
          <div className="validation-header">
            <span className="validation-icon">🧠</span>
            <span>Dinamična LLM validacija</span>
          </div>
          <div className="validation-content">
            <p>Rezultati so bili analizirani z naprednim LLM sistemom za boljšo natančnost</p>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-avatar">🧠</div>
            <h3>Pozdravljeni v revolucionalni slovenščini grocery intelligence!</h3>
            <p>Zdaj z <strong>dinamično LLM validacijo</strong> - brez trdno kodiranih pravil!</p>
            
            <div className="dynamic-validation-highlight">
              <h4>🚀 Revolucionarne izboljšave:</h4>
              <div className="validation-features">
                <div className="validation-feature">
                  <span className="feature-icon">🧠</span>
                  <div>
                    <strong>Dinamična LLM analiza</strong>
                    <p>Inteligentno razumevanje vaših potreb brez vnaprej določenih pravil</p>
                  </div>
                </div>
                <div className="validation-feature">
                  <span className="feature-icon">📊</span>
                  <div>
                    <strong>Analiza vsebine baze</strong>
                    <p>Najprej analizira, kaj je dejansko v bazi, nato filtrira</p>
                  </div>
                </div>
                <div className="validation-feature">
                  <span className="feature-icon">🎯</span>
                  <div>
                    <strong>Kontekstno razumevanje</strong>
                    <p>Razume namere uporabnika, ne le dobesednih besed</p>
                  </div>
                </div>
                <div className="validation-feature">
                  <span className="feature-icon">💡</span>
                  <div>
                    <strong>Pametni predlogi</strong>
                    <p>Inteligentne alternative na osnovi dejanske vsebine baze</p>
                  </div>
                </div>
                <div className="validation-feature">
                  <span className="feature-icon">📈</span>
                  <div>
                    <strong>Transparentnost validacije</strong>
                    <p>Pojasnilo zakaj so bili izdelki vključeni ali izključeni</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="capabilities-showcase">
              <h4>🎯 Kaj znoja nova tehnologija:</h4>
              <div className="capability-examples">
                <div className="capability-example">
                  <div className="example-search">"mleko"</div>
                  <div className="example-arrow">→</div>
                  <div className="example-result">Analizira bazo, razume da potrebujete mleko, filtrira čokolade</div>
                </div>
                <div className="capability-example">
                  <div className="example-search">"kruh"</div>
                  <div className="example-arrow">→</div>
                  <div className="example-result">Razlikuje med kruhom in drobtinami na osnovi konteksta</div>
                </div>
                <div className="capability-example">
                  <div className="example-search">Ni zadetkov</div>
                  <div className="example-arrow">→</div>
                  <div className="example-result">Generira pametne predloge na osnovi dostopnih izdelkov</div>
                </div>
              </div>
            </div>
            
            <div className="welcome-features">
              <div className="feature-grid">
                <div className="feature-item">
                  <span className="feature-icon">🧠</span>
                  <h4>LLM Validacija</h4>
                  <p>Dinamično razumevanje</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">📊</span>
                  <h4>Produkt Insights</h4>
                  <p>Podrobne analize z AI</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">💡</span>
                  <h4>Pametni predlogi</h4>
                  <p>Na osnovi vsebine baze</p>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🎯</span>
                  <h4>Kontekstna točnost</h4>
                  <p>Razume vaše potrebe</p>
                </div>
              </div>
            </div>

            <div className="quick-questions">
              <h4>Preizkusite nova zmogljivosti:</h4>
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
              {msg.role === 'user' ? '👤' : '🧠'}
            </div>
            <div className="message-bubble">
              <div className="message-content">
                {msg.content}
                
                {/* Enhanced function usage display */}
                {msg.function_used && (
                  <div className="function-used">
                    <span className="function-icon">⚡</span>
                    Funkcija: {msg.function_used}
                    {msg.dynamic_validation && (
                      <span className="validation-badge dynamic">
                        <span className="validation-icon">🧠</span>
                        Dinamična LLM validacija
                      </span>
                    )}
                  </div>
                )}
                
                {/* Dynamic validation information */}
                {renderDynamicValidationInfo(msg)}
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
            <div className="message-avatar">🧠</div>
            <div className="message-bubble">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <div className="typing-text">LLM analizira bazo podatkov in validira rezultate...</div>
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
            placeholder="Vprašajte o cenah, analizah ali načrtovanju... Nova LLM validacija bo razumela vaš namen!"
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
                {loading ? '⏳' : '🧠'}
              </span>
              Pošlji
            </button>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .dynamic-validation-highlight {
          background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
          border: 1px solid rgba(59, 130, 246, 0.2);
          border-radius: 16px;
          padding: 25px;
          margin: 30px 0;
          backdrop-filter: blur(10px);
        }
        
        .dynamic-validation-highlight h4 {
          color: var(--text-primary);
          margin-bottom: 20px;
          font-size: 1.2em;
          text-align: center;
        }
        
        .capabilities-showcase {
          background: rgba(16, 185, 129, 0.1);
          border: 1px solid rgba(16, 185, 129, 0.2);
          border-radius: 16px;
          padding: 25px;
          margin: 30px 0;
        }
        
        .capabilities-showcase h4 {
          color: var(--text-primary);
          margin-bottom: 20px;
          font-size: 1.2em;
          text-align: center;
        }
        
        .capability-examples {
          display: grid;
          gap: 15px;
        }
        
        .capability-example {
          display: grid;
          grid-template-columns: 1fr auto 2fr;
          align-items: center;
          gap: 15px;
          padding: 15px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .example-search {
          font-family: 'Courier New', monospace;
          background: rgba(59, 130, 246, 0.2);
          padding: 8px 12px;
          border-radius: 8px;
          color: var(--text-primary);
          font-weight: 600;
          text-align: center;
        }
        
        .example-arrow {
          font-size: 1.5em;
          color: var(--text-secondary);
        }
        
        .example-result {
          color: var(--text-secondary);
          font-size: 0.9em;
          line-height: 1.4;
        }
        
        .validation-info.dynamic-suggestions {
          background: rgba(59, 130, 246, 0.1);
          border-color: rgba(59, 130, 246, 0.3);
        }
        
        .validation-info.dynamic-success {
          background: rgba(16, 185, 129, 0.1);
          border-color: rgba(16, 185, 129, 0.3);
        }
        
        .validation-info.insights {
          background: rgba(147, 51, 234, 0.1);
          border-color: rgba(147, 51, 234, 0.3);
        }
        
        .validation-info.dynamic-generic {
          background: rgba(99, 102, 241, 0.1);
          border-color: rgba(99, 102, 241, 0.3);
        }
        
        .validation-reasoning {
          margin: 10px 0;
          padding: 12px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          border-left: 3px solid rgba(59, 130, 246, 0.5);
          font-size: 0.9em;
          line-height: 1.4;
        }
        
        .validation-stats {
          margin-top: 15px;
        }
        
        .confidence-score {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
        }
        
        .stat-label {
          font-size: 0.9em;
          color: var(--text-secondary);
          font-weight: 500;
          min-width: 120px;
        }
        
        .confidence-bar {
          flex: 1;
          height: 8px;
          background: rgba(0, 0, 0, 0.1);
          border-radius: 4px;
          overflow: hidden;
        }
        
        .confidence-fill {
          height: 100%;
          transition: width 0.5s ease;
          border-radius: 4px;
        }
        
        .confidence-value {
          font-size: 0.9em;
          font-weight: 600;
          color: var(--text-primary);
          min-width: 40px;
          text-align: right;
        }
        
        .smart-suggestions {
          margin-top: 15px;
          padding-top: 15px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .insights-summary {
          background: rgba(147, 51, 234, 0.1);
          padding: 12px;
          border-radius: 8px;
          margin-bottom: 15px;
          border-left: 3px solid rgba(147, 51, 234, 0.5);
        }
        
        .price-insights, .ai-recommendations {
          margin: 15px 0;
        }
        
        .price-insights h5, .ai-recommendations h5 {
          color: var(--text-primary);
          margin-bottom: 10px;
          font-size: 1em;
        }
        
        .price-insights ul, .ai-recommendations ul {
          margin: 8px 0;
          padding-left: 20px;
        }
        
        .price-insights li, .ai-recommendations li {
          color: var(--text-secondary);
          margin: 5px 0;
          font-size: 0.9em;
        }
        
        .validation-badge.dynamic {
          background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(16, 185, 129, 0.2));
          border: 1px solid rgba(59, 130, 246, 0.3);
          color: var(--text-primary);
        }
        
        .validation-badge.dynamic .validation-icon {
          font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
          .capability-example {
            grid-template-columns: 1fr;
            text-align: center;
            gap: 10px;
          }
          
          .example-arrow {
            display: none;
          }
          
          .confidence-score {
            flex-direction: column;
            align-items: stretch;
            gap: 8px;
          }
          
          .stat-label {
            min-width: auto;
            text-align: center;
          }
          
          .confidence-value {
            text-align: center;
          }
        }
      `}</style>
    </div>
  );
};

export default ChatAssistant;