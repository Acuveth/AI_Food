// App.js - Updated with Slovenian language support
import React, { useState, useRef, useEffect } from 'react';
import './css/index.css';

// Import components
import ApiService from './services/ApiService';
import WelcomeMessage from './components/WelcomeMessage';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);
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

  const sendMessage = async (message = inputValue) => {
    if (!message.trim()) return;

    const userMessage = { 
      role: 'user', 
      content: message,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await ApiService.sendIntelligentRequest(message);
      
      if (response.success) {
        const botMessage = {
          role: 'assistant',
          content: response.message || 'Tukaj so vaši rezultati:',
          data: response.data,
          intent: response.intent,
          approach: response.approach,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorMessage = {
          role: 'assistant',
          content: response.message || 'Oprostite, ne morem obdelati vaše zahteve.',
          error: true,
          data: response.data,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Napaka pri povezavi: ${error.message}`,
        error: true,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleMealSelect = async (meal) => {
    // Show generic loading message immediately in Slovenian
    const loadingMessage = {
      role: 'assistant',
      content: `Iščem cene živil...`,
      timestamp: new Date(),
      isLoading: true
    };
    
    setMessages(prev => [...prev, loadingMessage]);
    
    try {
      const response = await ApiService.analyzeMealGrocery(meal);
      
      // Remove the loading message and add the actual grocery analysis
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.isLoading);
        
        if (response.success) {
          const groceryMessage = {
            role: 'assistant',
            content: `Tukaj je analiza stroškov nakupovanja za ${meal.title}:`,
            data: response.data,
            intent: 'meal_grocery_analysis',
            approach: 'meal_grocery_analysis',
            timestamp: new Date()
          };
          
          return [...filtered, groceryMessage];
        } else {
          const errorMessage = {
            role: 'assistant',
            content: `Analiza stroškov nakupovanja za ${meal.title} ni uspela. ${response.message || ''}`,
            error: true,
            timestamp: new Date()
          };
          
          return [...filtered, errorMessage];
        }
      });
    } catch (error) {
      // Remove loading message and show error
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.isLoading);
        const errorMessage = {
          role: 'assistant',
          content: `Napaka pri analizi stroškov nakupovanja za ${meal.title}: ${error.message}`,
          error: true,
          timestamp: new Date()
        };
        
        return [...filtered, errorMessage];
      });
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🛒 Pameten Nakup</h1>
        <p>Inteligentno nakupovanje živil za Slovenijo - Najdite akcije, primerjajte cene, odkrijte recepte</p>
      </header>

      <main className="main-content">
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 && (
              <WelcomeMessage onSuggestionClick={sendMessage} />
            )}
            
            {messages.map((msg, index) => (
              <div key={index}>
                {msg.isLoading ? (
                  <div className="message assistant">
                    <div className="message-avatar">🛒</div>
                    <div className="message-bubble">
                      <div className="message-content">
                        <div className="typing-indicator">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                        <p style={{ marginTop: '10px' }}>{msg.content}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <ChatMessage
                    msg={msg}
                    onMealSelect={handleMealSelect}
                    onSuggestionClick={sendMessage}
                  />
                )}
              </div>
            ))}
            
            {loading && (
              <div className="message assistant">
                <div className="message-avatar">🛒</div>
                <div className="message-bubble">
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <p style={{ marginTop: '10px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                      Razmišljam...
                    </p>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <ChatInput
            inputValue={inputValue}
            setInputValue={setInputValue}
            onSendMessage={() => sendMessage()}
            loading={loading}
            inputRef={inputRef}
          />
        </div>
      </main>

      <footer className="app-footer">
        <p>© 2024 Pameten Nakup - Optimizirana arhitektura za slovenskega uporabnika</p>
      </footer>
    </div>
  );
}

export default App;