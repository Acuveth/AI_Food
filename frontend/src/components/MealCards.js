// components/MealCards.js - Enhanced with inline recipe display
import React, { useState } from 'react';

const MealCards = ({ meals, onMealSelect }) => {
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [showRecipe, setShowRecipe] = useState(false);

  const handleMealClick = async (meal) => {
    setSelectedMeal(meal);
    setShowRecipe(true);
    
    // Scroll to recipe section
    setTimeout(() => {
      const recipeElement = document.getElementById('recipe-display');
      if (recipeElement) {
        recipeElement.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };

  const handleBackToMeals = () => {
    setShowRecipe(false);
    setSelectedMeal(null);
  };

  const formatCalories = (nutrition) => {
    if (!nutrition || !nutrition.calories) return 'N/A';
    return `${Math.round(nutrition.calories)} cal`;
  };

  const estimateBasicPrice = (meal) => {
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
      {/* Meal Cards Grid */}
      <div className="meal-cards-grid">
        {meals.map((meal, index) => (
          <div 
            key={meal.id || index} 
            className={`meal-card ${selectedMeal?.id === meal.id ? 'selected' : ''}`}
            onClick={() => handleMealClick(meal)}
          >
            <div className="meal-image">
              {meal.image_url ? (
                <img 
                  src={meal.image_url} 
                  alt={meal.title}
                  onError={(e) => {
                    e.target.src = '/placeholder-meal.jpg';
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
                  {selectedMeal?.id === meal.id ? 'View Recipe' : 'Select Recipe'}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Inline Recipe Display */}
      {showRecipe && selectedMeal && (
        <div id="recipe-display" className="recipe-display-container">
          <div className="recipe-header">
            <button className="back-to-meals-btn" onClick={handleBackToMeals}>
              ← Back to Meal Options
            </button>
            <h2>{selectedMeal.title}</h2>
          </div>
          
          <div className="recipe-content">
            <div className="recipe-left">
              {/* Large Recipe Image */}
              <div className="recipe-image-large">
                {selectedMeal.image_url ? (
                  <img src={selectedMeal.image_url} alt={selectedMeal.title} />
                ) : (
                  <div className="recipe-image-placeholder-large">🍽️</div>
                )}
              </div>
              
              {/* Recipe Overview */}
              <div className="recipe-overview">
                <h3>Recipe Overview</h3>
                <div className="overview-grid">
                  <div className="overview-item">
                    <span className="overview-label">Prep Time:</span>
                    <span>{selectedMeal.prep_time || 0} minutes</span>
                  </div>
                  <div className="overview-item">
                    <span className="overview-label">Cook Time:</span>
                    <span>{selectedMeal.cook_time || 0} minutes</span>
                  </div>
                  <div className="overview-item">
                    <span className="overview-label">Total Time:</span>
                    <span>{(selectedMeal.prep_time || 0) + (selectedMeal.cook_time || 0)} minutes</span>
                  </div>
                  <div className="overview-item">
                    <span className="overview-label">Servings:</span>
                    <span>{selectedMeal.servings || 2}</span>
                  </div>
                  <div className="overview-item">
                    <span className="overview-label">Difficulty:</span>
                    <span className="difficulty-badge">{selectedMeal.difficulty || 'Medium'}</span>
                  </div>
                  <div className="overview-item">
                    <span className="overview-label">Cuisine:</span>
                    <span>{selectedMeal.cuisine_type || 'International'}</span>
                  </div>
                </div>
              </div>
              
              {/* Nutrition Information */}
              {selectedMeal.nutrition && Object.keys(selectedMeal.nutrition).length > 0 && (
                <div className="nutrition-section">
                  <h3>Nutrition Information</h3>
                  <div className="nutrition-grid">
                    {selectedMeal.nutrition.calories && (
                      <div className="nutrition-item">
                        <span className="nutrition-icon">🔥</span>
                        <span className="nutrition-label">Calories</span>
                        <span className="nutrition-value">{Math.round(selectedMeal.nutrition.calories)}</span>
                      </div>
                    )}
                    {selectedMeal.nutrition.protein && (
                      <div className="nutrition-item">
                        <span className="nutrition-icon">💪</span>
                        <span className="nutrition-label">Protein</span>
                        <span className="nutrition-value">{selectedMeal.nutrition.protein}</span>
                      </div>
                    )}
                    {selectedMeal.nutrition.carbs && (
                      <div className="nutrition-item">
                        <span className="nutrition-icon">🌾</span>
                        <span className="nutrition-label">Carbs</span>
                        <span className="nutrition-value">{selectedMeal.nutrition.carbs}</span>
                      </div>
                    )}
                    {selectedMeal.nutrition.fat && (
                      <div className="nutrition-item">
                        <span className="nutrition-icon">🥑</span>
                        <span className="nutrition-label">Fat</span>
                        <span className="nutrition-value">{selectedMeal.nutrition.fat}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            <div className="recipe-right">
              {/* Ingredients List */}
              <div className="ingredients-section">
                <h3>🛒 Ingredients</h3>
                {selectedMeal.ingredients && selectedMeal.ingredients.length > 0 ? (
                  <div className="ingredients-list">
                    {selectedMeal.ingredients.map((ingredient, index) => (
                      <div key={index} className="ingredient-item">
                        <span className="ingredient-amount">
                          {ingredient.amount && ingredient.unit 
                            ? `${ingredient.amount} ${ingredient.unit}` 
                            : ingredient.original || ingredient.name}
                        </span>
                        <span className="ingredient-name">
                          {ingredient.name || ''}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-ingredients">Ingredients list not available for this recipe.</p>
                )}
              </div>
              
              {/* Diet Labels & Allergen Info */}
              {(selectedMeal.diet_labels?.length > 0 || selectedMeal.allergen_info?.length > 0) && (
                <div className="dietary-info">
                  {selectedMeal.diet_labels?.length > 0 && (
                    <div className="diet-labels">
                      <h4>🥗 Dietary Information</h4>
                      <div className="labels-list">
                        {selectedMeal.diet_labels.map((label, index) => (
                          <span key={index} className="diet-label">{label}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {selectedMeal.allergen_info?.length > 0 && (
                    <div className="allergen-info">
                      <h4>⚠️ Allergen Information</h4>
                      <div className="allergen-list">
                        {selectedMeal.allergen_info.map((allergen, index) => (
                          <span key={index} className="allergen-tag">{allergen}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Cooking Instructions */}
              <div className="instructions-section">
                <h3>👨‍🍳 Cooking Instructions</h3>
                {selectedMeal.instructions && selectedMeal.instructions.length > 0 ? (
                  <div className="instructions-list">
                    {selectedMeal.instructions.map((instruction, index) => (
                      <div key={index} className="instruction-step">
                        <div className="step-number">{index + 1}</div>
                        <div className="step-content">{instruction}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-instructions">
                    <p>Detailed instructions are not available for this recipe.</p>
                    {selectedMeal.recipe_url && (
                      <p>
                        <a 
                          href={selectedMeal.recipe_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="external-recipe-link"
                        >
                          View full recipe on original site →
                        </a>
                      </p>
                    )}
                  </div>
                )}
              </div>
              
              {/* Action Buttons */}
              <div className="recipe-actions">
                {selectedMeal.recipe_url && (
                  <a 
                    href={selectedMeal.recipe_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="action-btn external-link-btn"
                  >
                    🔗 View Original Recipe
                  </a>
                )}
                
                <button 
                  className="action-btn grocery-btn"
                  onClick={() => onMealSelect && onMealSelect(selectedMeal, null)}
                >
                  🛒 Get Grocery List
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Keep the existing MealDetailView for grocery integration
const MealDetailView = ({ meal, groceryDetails, onBack }) => {
  if (!meal) return null;

  const formatPrice = (price) => `€${(price || 0).toFixed(2)}`;

  return (
    <div className="meal-detail-container">
      <div className="meal-detail-header">
        <button className="back-button" onClick={onBack}>
          ← Back to Recipe
        </button>
        <h2>🛒 Grocery Shopping for: {meal.title}</h2>
      </div>
      
      <div className="meal-detail-content">
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
      </div>
    </div>
  );
};

export default MealCards;
export { MealDetailView };