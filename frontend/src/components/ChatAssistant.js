// components/ChatAssistant.js - Updated with inline recipe support
import React, { useState, useRef, useEffect } from 'react';
import ApiService from '../services/api';
import MealCards, { MealDetailView } from './MealCards';

const ChatAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [mealDetails, setMealDetails] = useState(null);
  const [showGroceryDetails, setShowGroceryDetails] = useState(false);
  const [loadingGrocery, setLoadingGrocery] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, showGroceryDetails]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const quickQuestions = [
    "Find Italian lunch recipes",
    "Healthy breakfast ideas", 
    "Quick dinner for 2 people",
    "Vegetarian meal options"
  ];

  // Check if a message contains meal search results
  const containsMealSearch = (content) => {
    const mealKeywords = ['recipe', 'meal', 'lunch', 'dinner', 'breakfast', 'cook', 'ingredient'];
    return mealKeywords.some(keyword => content.toLowerCase().includes(keyword));
  };

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
      // Check if this is likely a meal search request
      if (containsMealSearch(message)) {
        // Try meal search first
        try {
          const mealResponse = await ApiService.searchMeals(message);
          
          if (mealResponse.success && mealResponse.data.meals?.length > 0) {
            const botMessage = { 
              role: 'assistant', 
              content: mealResponse.data.presentation?.summary || `Found ${mealResponse.data.meals.length} meal options for you!`,
              timestamp: new Date(),
              meals: mealResponse.data.meals,
              isMealSearch: true
            };
            setMessages(prev => [...prev, botMessage]);
            setLoading(false);
            return;
          }
        } catch (mealError) {
          console.log('Meal search failed, falling back to general chat:', mealError);
        }
      }

      // Fallback to general chat
      const response = await ApiService.sendChatMessage(message);
      
      if (response.success) {
        const botMessage = { 
          role: 'assistant', 
          content: response.data.response,
          timestamp: new Date(),
          isMarkdown: true
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

  // NEW: Handle when user clicks "Get Grocery List" button in recipe display
  const handleGroceryRequest = async (meal) => {
    setSelectedMeal(meal);
    setLoadingGrocery(true);
    
    try {
      // Get detailed grocery information for selected meal
      const response = await ApiService.getMealDetails(meal.id, meal);
      
      if (response.success) {
        setMealDetails(response.data);
        setShowGroceryDetails(true);
      } else {
        console.error('Failed to get meal details:', response.error);
        // Show error message to user
        alert('Failed to get grocery details. Please try again.');
      }
    } catch (error) {
      console.error('Error getting meal details:', error);
      alert('Failed to get grocery details. Please try again.');
    } finally {
      setLoadingGrocery(false);
    }
  };

  const handleBackToRecipe = () => {
    setShowGroceryDetails(false);
    setMealDetails(null);
    // Keep selectedMeal so recipe stays visible
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
    // Show grocery detail view if selected
    if (showGroceryDetails && selectedMeal && mealDetails) {
      return (
        <MealDetailView 
          meal={selectedMeal}
          groceryDetails={mealDetails.grocery_details}
          onBack={handleBackToRecipe}
        />
      );
    }

    // Show meal cards with inline recipe display
    if (msg.isMealSearch && msg.meals) {
      return (
        <div className="meal-search-result">
          <div className="message-content">
            {msg.content}
          </div>
          <MealCards 
            meals={msg.meals} 
            onMealSelect={handleGroceryRequest} // This now triggers grocery request
          />
          
          {/* Loading overlay for grocery details */}
          {loadingGrocery && (
            <div className="loading-overlay">
              <div className="loading-spinner"></div>
              <p>Getting grocery details...</p>
            </div>
          )}
        </div>
      );
    }

    // Regular markdown content
    if (msg.isMarkdown) {
      const htmlContent = processMarkdown(msg.content);
      return (
        <div 
          className="message-content markdown-content"
          dangerouslySetInnerHTML={{ __html: htmlContent }}
        />
      );
    } 

    // Regular text content
    return (
      <div className="message-content">
        {msg.content}
      </div>
    );
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && !showGroceryDetails && (
          <div className="welcome-message">
            <div className="welcome-avatar">🛒</div>
            <h3>Slovenian Grocery & Meal Assistant</h3>
            <p>Find the best prices and discover delicious recipes with complete cooking instructions</p>
            
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
        
        {!showGroceryDetails && messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
            <div className={`message-bubble ${msg.isMealSearch ? 'full-width' : ''}`}>
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

        {showGroceryDetails && (
          <div className="message assistant">
            <div className="message-bubble full-width">
              {renderMessageContent({ isMealSearch: false })}
            </div>
            <div className="message-avatar">🛒</div>
          </div>
        )}
        
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
            <div className="message-avatar">🛒</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {!showGroceryDetails && (
        <div className="chat-input">
          <div className="input-container">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about meals, recipes, or grocery prices..."
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
      )}
    </div>
  );
};

export default ChatAssistant;