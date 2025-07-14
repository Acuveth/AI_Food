// components/RecipeDisplay.js - Immediate recipe display
import React from 'react';

const RecipeDisplay = ({ meal, onBackToMeals, loadingGrocery }) => {
  if (!meal) return null;

  return (
    <div className="recipe-display-container">
      <div className="recipe-header">
        <button className="back-to-meals-btn" onClick={onBackToMeals}>
          ← Back to Meal Options
        </button>
        <h2>{meal.title}</h2>
      </div>
      
      <div className="recipe-content">
        <div className="recipe-left">
          {/* Large Recipe Image */}
          <div className="recipe-image-large">
            {meal.image_url ? (
              <img src={meal.image_url} alt={meal.title} />
            ) : (
              <div className="recipe-image-placeholder-large">🍽️</div>
            )}
          </div>
          
          {/* Recipe Overview */}
          <div className="recipe-overview">
            <h3>📋 Recipe Overview</h3>
            <div className="overview-grid">
              <div className="overview-item">
                <span className="overview-label">Prep Time:</span>
                <span>{meal.prep_time || 0} minutes</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Cook Time:</span>
                <span>{meal.cook_time || 0} minutes</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Total Time:</span>
                <span>{(meal.prep_time || 0) + (meal.cook_time || 0)} minutes</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Servings:</span>
                <span>{meal.servings || 2}</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Difficulty:</span>
                <span className="difficulty-badge">{meal.difficulty || 'Medium'}</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Cuisine:</span>
                <span>{meal.cuisine_type || 'International'}</span>
              </div>
            </div>
          </div>
          
          {/* Nutrition Information */}
          {meal.nutrition && Object.keys(meal.nutrition).length > 0 && (
            <div className="nutrition-section">
              <h3>🍎 Nutrition Information</h3>
              <div className="nutrition-grid">
                {meal.nutrition.calories && (
                  <div className="nutrition-item">
                    <span className="nutrition-icon">🔥</span>
                    <span className="nutrition-label">Calories</span>
                    <span className="nutrition-value">{Math.round(meal.nutrition.calories)}</span>
                  </div>
                )}
                {meal.nutrition.protein && (
                  <div className="nutrition-item">
                    <span className="nutrition-icon">💪</span>
                    <span className="nutrition-label">Protein</span>
                    <span className="nutrition-value">{meal.nutrition.protein}</span>
                  </div>
                )}
                {meal.nutrition.carbs && (
                  <div className="nutrition-item">
                    <span className="nutrition-icon">🌾</span>
                    <span className="nutrition-label">Carbs</span>
                    <span className="nutrition-value">{meal.nutrition.carbs}</span>
                  </div>
                )}
                {meal.nutrition.fat && (
                  <div className="nutrition-item">
                    <span className="nutrition-icon">🥑</span>
                    <span className="nutrition-label">Fat</span>
                    <span className="nutrition-value">{meal.nutrition.fat}</span>
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
            {meal.ingredients && meal.ingredients.length > 0 ? (
              <div className="ingredients-list-no-scroll">
                {meal.ingredients.map((ingredient, index) => (
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
          {(meal.diet_labels?.length > 0 || meal.allergen_info?.length > 0) && (
            <div className="dietary-info">
              {meal.diet_labels?.length > 0 && (
                <div className="diet-labels">
                  <h4>🥗 Dietary Information</h4>
                  <div className="labels-list">
                    {meal.diet_labels.map((label, index) => (
                      <span key={index} className="diet-label">{label}</span>
                    ))}
                  </div>
                </div>
              )}
              
              {meal.allergen_info?.length > 0 && (
                <div className="allergen-info">
                  <h4>⚠️ Allergen Information</h4>
                  <div className="allergen-list">
                    {meal.allergen_info.map((allergen, index) => (
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
            {meal.instructions && meal.instructions.length > 0 ? (
              <div className="instructions-list">
                {meal.instructions.map((instruction, index) => (
                  <div key={index} className="instruction-step">
                    <div className="step-number">{index + 1}</div>
                    <div className="step-content">{instruction}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-instructions">
                <p>Detailed instructions are not available for this recipe.</p>
                {meal.recipe_url && (
                  <p>
                    <a 
                      href={meal.recipe_url} 
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
            {meal.recipe_url && (
              <a 
                href={meal.recipe_url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="action-btn external-link-btn"
              >
                🔗 View Original Recipe
              </a>
            )}
            
            <div className="grocery-status">
              {loadingGrocery ? (
                <div className="grocery-loading-status">
                  <div className="mini-spinner"></div>
                  <span>🛒 Finding grocery prices...</span>
                </div>
              ) : (
                <div className="grocery-ready-status">
                  <span>✅ Grocery prices will appear below</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecipeDisplay;