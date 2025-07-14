// App.js - Updated with immediate recipe display and background grocery analysis
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
          content: response.message || 'Here are your results:',
          data: response.data,
          intent: response.intent,
          approach: response.approach,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorMessage = {
          role: 'assistant',
          content: response.message || 'Sorry, I could not process your request.',
          error: true,
          data: response.data,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Connection error: ${error.message}`,
        error: true,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleMealSelect = async (meal) => {
    // Show generic loading message immediately
    const loadingMessage = {
      role: 'assistant',
      content: `Finding grocery prices...`,
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
            content: `Here's the grocery cost analysis for ${meal.title}:`,
            data: response.data,
            intent: 'meal_grocery_analysis',
            approach: 'meal_grocery_analysis',
            timestamp: new Date()
          };
          
          return [...filtered, groceryMessage];
        } else {
          const errorMessage = {
            role: 'assistant',
            content: `Failed to analyze grocery costs for ${meal.title}. ${response.message || ''}`,
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
          content: `Error analyzing grocery costs for ${meal.title}: ${error.message}`,
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
        <h1>🛒 Grocery Intelligence</h1>
        <p>Smart grocery shopping for Slovenia - Find deals, compare prices, discover meals</p>
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
        <p>© 2024 Grocery Intelligence - Streamlined Architecture</p>
      </footer>
    </div>
  );
}

export default App;