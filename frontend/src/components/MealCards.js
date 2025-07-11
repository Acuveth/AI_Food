// components/MealCards.js
import React, { useState } from 'react';
import ApiService from '../services/api';

const MealCards = ({ meals, onMealSelect }) => {
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [mealDetails, setMealDetails] = useState(null);

  const handleMealClick = async (meal) => {
    setSelectedMeal(meal);
    setLoadingDetails(true);
    
    try {
      // Get detailed grocery information for selected meal
      const response = await ApiService.getMealDetails(meal.id, meal);
      
      if (response.success) {
        setMealDetails(response.data);
        onMealSelect(meal, response.data);
      } else {
        console.error('Failed to get meal details:', response.error);
      }
    } catch (error) {
      console.error('Error getting meal details:', error);
    } finally {
      setLoadingDetails(false);
    }
  };

  const formatCalories = (nutrition) => {
    if (!nutrition || !nutrition.calories) return 'N/A';
    return `${Math.round(nutrition.calories)} cal`;
  };

  const estimateBasicPrice = (meal) => {
    // Simple price estimation based on ingredients count and complexity
    const basePrice = 2.50;
    const ingredientCount = meal.ingredients?.length || 5;
    const complexityMultiplier = meal.difficulty === 'easy' ? 0.8 : meal.difficulty === 'hard' ? 1.3 : 1.0;
    
    return (basePrice + (ingredientCount * 0.30) * complexityMultiplier).toFixed(2);
  };

  if (!meals || meals.length === 0) {
    return (
      <div className="no-meals">
        <h3>No meals found</h3>
        <p>Try different search terms or check your API configuration</p>
      </div>
    );
  }

  return (
    <div className="meal-cards-container">
      <div className="meal-cards-grid">
        {meals.map((meal, index) => (
          <div 
            key={meal.id || index} 
            className="meal-card"
            onClick={() => handleMealClick(meal)}
          >
            <div className="meal-image">
              {meal.image_url ? (
                <img 
                  src={meal.image_url} 
                  alt={meal.title}
                  onError={(e) => {
                    e.target.src = '/placeholder-meal.jpg'; // Fallback image
                  }}
                />
              ) : (
                <div className="meal-image-placeholder">
                  <span>🍽️</span>
                </div>
              )}
            </div>
            
            <div className="meal-card-content">
              <h3 className="meal-title">{meal.title}</h3>
              
              <div className="meal-info">
                <div className="meal-stat">
                  <span className="stat-icon">🔥</span>
                  <span>{formatCalories(meal.nutrition)}</span>
                </div>
                
                <div className="meal-stat">
                  <span className="stat-icon">⏱️</span>
                  <span>{(meal.prep_time || 0) + (meal.cook_time || 0)} min</span>
                </div>
                
                <div className="meal-stat">
                  <span className="stat-icon">👥</span>
                  <span>{meal.servings || 2} servings</span>
                </div>
              </div>
              
              <div className="meal-price">
                <span className="price-label">Est. Cost:</span>
                <span className="price-value">€{estimateBasicPrice(meal)}</span>
              </div>
              
              {meal.cuisine_type && (
                <div className="meal-cuisine">
                  {meal.cuisine_type}
                </div>
              )}
              
              <div className="meal-card-footer">
                <button className="select-meal-btn">
                  Select Meal
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {loadingDetails && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Getting grocery details...</p>
        </div>
      )}
    </div>
  );
};

// components/MealDetailView.js  
const MealDetailView = ({ meal, groceryDetails, onBack }) => {
  if (!meal) return null;

  const formatPrice = (price) => `€${(price || 0).toFixed(2)}`;

  return (
    <div className="meal-detail-container">
      <div className="meal-detail-header">
        <button className="back-button" onClick={onBack}>
          ← Back to Meals
        </button>
        <h2>{meal.title}</h2>
      </div>
      
      <div className="meal-detail-content">
        <div className="meal-detail-left">
          <div className="meal-image-large">
            {meal.image_url ? (
              <img src={meal.image_url} alt={meal.title} />
            ) : (
              <div className="meal-image-placeholder-large">🍽️</div>
            )}
          </div>
          
          <div className="meal-info-detailed">
            <h3>Meal Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Prep Time:</span>
                <span>{meal.prep_time || 0} minutes</span>
              </div>
              <div className="info-item">
                <span className="info-label">Cook Time:</span>
                <span>{meal.cook_time || 0} minutes</span>
              </div>
              <div className="info-item">
                <span className="info-label">Servings:</span>
                <span>{meal.servings || 2}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Difficulty:</span>
                <span>{meal.difficulty || 'Medium'}</span>
              </div>
            </div>
          </div>
          
          {meal.nutrition && Object.keys(meal.nutrition).length > 0 && (
            <div className="nutrition-info">
              <h3>Nutrition</h3>
              <div className="nutrition-grid">
                {meal.nutrition.calories && (
                  <div className="nutrition-item">
                    <span className="nutrition-label">Calories:</span>
                    <span>{Math.round(meal.nutrition.calories)}</span>
                  </div>
                )}
                {meal.nutrition.protein && (
                  <div className="nutrition-item">
                    <span className="nutrition-label">Protein:</span>
                    <span>{meal.nutrition.protein}</span>
                  </div>
                )}
                {meal.nutrition.carbs && (
                  <div className="nutrition-item">
                    <span className="nutrition-label">Carbs:</span>
                    <span>{meal.nutrition.carbs}</span>
                  </div>
                )}
                {meal.nutrition.fat && (
                  <div className="nutrition-item">
                    <span className="nutrition-label">Fat:</span>
                    <span>{meal.nutrition.fat}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        <div className="meal-detail-right">
          <div className="grocery-section">
            <h3>🛒 Slovenian Grocery Shopping</h3>
            
            {groceryDetails && groceryDetails.shopping_list?.length > 0 ? (
              <>
                <div className="total-cost">
                  <span className="cost-label">Total Estimated Cost:</span>
                  <span className="cost-value">{formatPrice(groceryDetails.estimated_cost)}</span>
                </div>
                
                <div className="available-stores">
                  <h4>Available in stores:</h4>
                  <div className="stores-list">
                    {groceryDetails.available_stores?.map((store, index) => (
                      <span key={index} className="store-badge">
                        {store.toUpperCase()}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="shopping-list">
                  <h4>Shopping List:</h4>
                  <div className="ingredients-list">
                    {groceryDetails.shopping_list.map((item, index) => (
                      <div key={index} className="ingredient-item">
                        <div className="ingredient-info">
                          <span className="ingredient-name">{item.ingredient}</span>
                          <span className="ingredient-slovenian">({item.slovenian_name})</span>
                          <span className="ingredient-amount">{item.needed_amount}</span>
                        </div>
                        <div className="ingredient-product">
                          <span className="product-name">{item.product?.product_name}</span>
                          <span className="product-store">{item.product?.store_name?.toUpperCase()}</span>
                          <span className="product-price">{formatPrice(item.estimated_cost)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="no-grocery-info">
                <p>Some ingredients may not be available in our Slovenian grocery database.</p>
                <p>You can still prepare this meal by shopping at local stores.</p>
              </div>
            )}
          </div>
          
          {meal.recipe_url && (
            <div className="recipe-link">
              <a 
                href={meal.recipe_url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="full-recipe-btn"
              >
                View Full Recipe →
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MealCards;
export { MealDetailView };