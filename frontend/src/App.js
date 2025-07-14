// Streamlined React Frontend - App.js
import React, { useState, useRef, useEffect } from 'react';
import './Streamlined.css';

// API Service
class ApiService {
  static BASE_URL = 'http://localhost:8000';

  static async request(endpoint, options = {}) {
    const url = `${this.BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  // Main intelligent request endpoint
  static async sendIntelligentRequest(input) {
    return this.request('/api/intelligent-request', {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
  }

  // Direct function calls
  static async getPromotions(searchFilter = null) {
    const endpoint = searchFilter 
      ? `/api/promotions/all?search=${encodeURIComponent(searchFilter)}`
      : '/api/promotions/all';
    return this.request(endpoint);
  }

  static async comparePrices(itemName) {
    return this.request(`/api/compare-prices/${encodeURIComponent(itemName)}`);
  }

  static async searchMeals(request) {
    return this.request('/api/search-meals', {
      method: 'POST',
      body: JSON.stringify({ request }),
    });
  }

  static async analyzeMealGrocery(mealData) {
    return this.request('/api/meal-grocery-analysis', {
      method: 'POST',
      body: JSON.stringify({ meal_data: mealData }),
    });
  }

  static async findMealsFromIngredients(ingredients) {
    return this.request('/api/meals-from-ingredients', {
      method: 'POST',
      body: JSON.stringify({ ingredients }),
    });
  }
}

// Main App Component
function App() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [showGroceryAnalysis, setShowGroceryAnalysis] = useState(false);
  const [groceryData, setGroceryData] = useState(null);
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

  const quickSuggestions = [
    "Find milk promotions",
    "Compare bread prices across stores",
    "Healthy Italian dinner recipes",
    "Meals I can make with chicken and rice",
    "Vegetarian lunch ideas",
    "Cheapest pasta options"
  ];

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
    setSelectedMeal(meal);
    setLoading(true);
    
    try {
      const response = await ApiService.analyzeMealGrocery(meal);
      
      if (response.success) {
        setGroceryData(response.data);
        setShowGroceryAnalysis(true);
        
        // Add grocery analysis message
        const groceryMessage = {
          role: 'assistant',
          content: `Here's the grocery cost analysis for ${meal.title}:`,
          data: response.data,
          intent: 'meal_grocery_analysis',
          approach: 'meal_grocery_analysis',
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, groceryMessage]);
      } else {
        alert('Failed to analyze grocery costs for this meal');
      }
    } catch (error) {
      alert('Error analyzing meal costs');
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

  const formatTime = (timestamp) => {
    return timestamp ? timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  };

  const renderMessageContent = (msg) => {
    const { intent, approach, data } = msg;

    // Handle different types of responses
    if (intent === 'FIND_PROMOTIONS' && data?.promotions) {
      return <PromotionResults data={data} />;
    }
    
    if (intent === 'COMPARE_ITEM_PRICES' && data?.results_by_store) {
      return <PriceComparisonResults data={data} />;
    }
    
    if (intent === 'SEARCH_MEALS' && data?.meals) {
      return <MealResults data={data} onMealSelect={handleMealSelect} />;
    }
    
    if (intent === 'REVERSE_MEAL_SEARCH' && data?.suggested_meals) {
      return <ReverseMealResults data={data} onMealSelect={handleMealSelect} />;
    }
    
    if (approach === 'meal_grocery_analysis' && data?.grocery_analysis) {
      return <GroceryAnalysisResults data={data} />;
    }
    
    if (intent === 'GENERAL_QUESTION' && data?.suggestions) {
      return <GeneralHelpResults data={data} onSuggestionClick={sendMessage} />;
    }
    
    if (approach === 'clarification_needed' && data?.clarification_questions) {
      return <ClarificationResults data={data} onSuggestionClick={sendMessage} />;
    }

    // Default text content
    return (
      <div className="message-content">
        {msg.content}
      </div>
    );
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
                        onClick={() => sendMessage(suggestion)}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🛒'}
                </div>
                <div className="message-bubble">
                  {renderMessageContent(msg)}
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
                  onClick={() => sendMessage()}
                  disabled={loading || !inputValue.trim()}
                >
                  {loading ? '⏳' : '→'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>© 2024 Grocery Intelligence - Streamlined Architecture</p>
      </footer>
    </div>
  );
}

// Promotion Results Component
const PromotionResults = ({ data }) => {
  const { promotions, analysis, category_breakdown, store_breakdown } = data;

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🏷️ Promotional Items Found</h3>
        <p>{data.summary}</p>
      </div>
      
      <div className="promotions-grid">
        {promotions.slice(0, 12).map((promo, index) => (
          <div key={index} className="promotion-card">
            <h4>{promo.product_name}</h4>
            <div className="price-info">
              <span className="current-price">€{promo.current_price?.toFixed(2)}</span>
              {promo.regular_price && (
                <span className="original-price">€{promo.regular_price?.toFixed(2)}</span>
              )}
            </div>
            <div className="store-info">{promo.store_name?.toUpperCase()}</div>
            <div className="discount-badge">
              {promo.discount_percentage}% OFF
            </div>
            {promo.deal_quality && (
              <div className={`deal-quality ${promo.deal_quality}`}>
                {promo.deal_quality} deal
              </div>
            )}
          </div>
        ))}
      </div>

      {analysis?.highlights && (
        <div className="analysis-summary">
          <h4>📊 Analysis Highlights</h4>
          <div className="highlights-grid">
            <div className="highlight-card">
              <strong>Best Discount</strong>
              <p>{analysis.highlights.best_discount.product} at {analysis.highlights.best_discount.store}</p>
              <span className="highlight-value">{analysis.highlights.best_discount.discount}% OFF</span>
            </div>
            <div className="highlight-card">
              <strong>Biggest Savings</strong>
              <p>{analysis.highlights.biggest_savings.product}</p>
              <span className="highlight-value">€{analysis.highlights.biggest_savings.savings}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Price Comparison Results Component
const PriceComparisonResults = ({ data }) => {
  const { results_by_store, price_analysis, best_deals, store_rankings } = data;

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🔍 Price Comparison Results</h3>
        <p>{data.summary}</p>
      </div>

      <div className="price-comparison-grid">
        {Object.entries(results_by_store).map(([store, storeData]) => (
          <div key={store} className={`store-card ${storeData.product_count === 0 ? 'no-products' : ''}`}>
            <h4>{storeData.store_name}</h4>
            
            {storeData.product_count > 0 ? (
              <>
                <div className="store-stats">
                  <p><strong>{storeData.product_count}</strong> products found</p>
                  <p>Avg: <strong>€{storeData.avg_price?.toFixed(2)}</strong></p>
                  {storeData.cheapest_product && (
                    <p>Cheapest: <strong>€{storeData.cheapest_product.current_price?.toFixed(2)}</strong></p>
                  )}
                </div>
                
                <div className="products-list">
                  {storeData.products.slice(0, 3).map((product, index) => (
                    <div key={index} className="product-item">
                      <span className="product-name">{product.product_name}</span>
                      <span className="product-price">€{product.current_price?.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="no-products-message">No products found in this store</p>
            )}
          </div>
        ))}
      </div>

      {price_analysis?.cheapest_option && (
        <div className="best-deal-highlight">
          <h4>🏆 Best Deal</h4>
          <div className="best-deal-card">
            <strong>{price_analysis.cheapest_option.product_name}</strong>
            <p>at {price_analysis.cheapest_option.store}</p>
            <span className="best-price">€{price_analysis.cheapest_option.price?.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Meal Results Component
const MealResults = ({ data, onMealSelect }) => {
  const { meals, request_analysis } = data;

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🍽️ Meal Suggestions</h3>
        <p>{data.summary}</p>
      </div>

      <div className="meals-grid">
        {meals.map((meal, index) => (
          <div key={index} className="meal-card" onClick={() => onMealSelect(meal)}>
            {meal.image_url && (
              <div className="meal-image">
                <img src={meal.image_url} alt={meal.title} />
              </div>
            )}
            
            <div className="meal-content">
              <h4>{meal.title}</h4>
              <p className="meal-description">{meal.description}</p>
              
              <div className="meal-info">
                <span>⏱️ {(meal.prep_time || 0) + (meal.cook_time || 0)} min</span>
                <span>👥 {meal.servings || 2} servings</span>
                {meal.cuisine_type && <span>🌍 {meal.cuisine_type}</span>}
              </div>
              
              {meal.diet_labels?.length > 0 && (
                <div className="diet-labels">
                  {meal.diet_labels.slice(0, 2).map((diet, i) => (
                    <span key={i} className="diet-label">{diet}</span>
                  ))}
                </div>
              )}
              
              <button className="select-meal-btn">
                View Recipe & Grocery Prices
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Reverse Meal Results Component
const ReverseMealResults = ({ data, onMealSelect }) => {
  const { suggested_meals, available_ingredients } = data;

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🥗 Meals You Can Make</h3>
        <p>Using your ingredients: {available_ingredients.join(', ')}</p>
      </div>

      <div className="meals-grid">
        {suggested_meals.map((meal, index) => (
          <div key={index} className="meal-card" onClick={() => onMealSelect(meal)}>
            {meal.image_url && (
              <div className="meal-image">
                <img src={meal.image_url} alt={meal.title} />
              </div>
            )}
            
            <div className="meal-content">
              <h4>{meal.title}</h4>
              <p className="meal-description">{meal.description}</p>
              
              <div className="ingredient-match">
                <span className="match-score">
                  Match: {((meal.ingredient_match_score || 0) * 100).toFixed(0)}%
                </span>
              </div>
              
              <button className="select-meal-btn">
                See Full Recipe & Costs
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Grocery Analysis Results Component
const GroceryAnalysisResults = ({ data }) => {
  const { meal, grocery_analysis } = data;
  const { store_analysis, combined_analysis, meal_statistics } = grocery_analysis;

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🛒 Grocery Cost Analysis: {meal.title}</h3>
        <p>{data.summary}</p>
      </div>

      <div className="grocery-analysis">
        <div className="cost-summary">
          <div className="total-cost-card">
            <h4>Total Estimated Cost</h4>
            <span className="total-amount">€{combined_analysis.total_cost?.toFixed(2)}</span>
            <p>Cost per serving: €{meal_statistics.cost_per_serving?.toFixed(2)}</p>
          </div>
        </div>

        <div className="store-comparison">
          <h4>Store-by-Store Comparison</h4>
          <div className="stores-grid">
            {Object.entries(store_analysis).map(([store, analysis]) => (
              <div key={store} className={`store-cost-card ${analysis.completeness === 100 ? 'best-option' : ''}`}>
                <h5>{analysis.store_name}</h5>
                <div className="cost-info">
                  <span className="store-total">€{analysis.total_cost?.toFixed(2)}</span>
                  <span className="completeness">{analysis.completeness?.toFixed(0)}% complete</span>
                </div>
                <p>{analysis.available_items}/{grocery_analysis.ingredient_results ? Object.keys(grocery_analysis.ingredient_results).length : 0} ingredients</p>
                {analysis.missing_items?.length > 0 && (
                  <div className="missing-items">
                    <small>Missing: {analysis.missing_items.slice(0, 2).join(', ')}</small>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="ingredients-breakdown">
          <h4>Ingredient Prices</h4>
          <div className="ingredients-list">
            {combined_analysis.item_details?.map((item, index) => (
              <div key={index} className="ingredient-row">
                <span className="ingredient-name">{item.ingredient}</span>
                <div className="ingredient-price">
                  {item.found ? (
                    <>
                      <span className="price">€{item.price?.toFixed(2)}</span>
                      <span className="store">{item.store?.toUpperCase()}</span>
                    </>
                  ) : (
                    <span className="not-found">Not found</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

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

export default App;