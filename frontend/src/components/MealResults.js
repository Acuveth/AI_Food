// components/MealResults.js - Updated for immediate recipe display
import React, { useState } from 'react';
import RecipeDisplay from './RecipeDisplay';

const MealResults = ({ data, onMealSelect }) => {
  const { meals, request_analysis } = data;
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [showRecipe, setShowRecipe] = useState(false);
  const [loadingGrocery, setLoadingGrocery] = useState(false);

  const handleMealClick = async (meal) => {
    setSelectedMeal(meal);
    setShowRecipe(true);
    setLoadingGrocery(true);
    
    // Immediately start grocery analysis
    await onMealSelect(meal);
    setLoadingGrocery(false);
    
    // Scroll to recipe section after a brief delay
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
    setLoadingGrocery(false);
  };

  // If showing recipe, display that instead of meal grid
  if (showRecipe && selectedMeal) {
    return (
      <div id="recipe-display" className="recipe-container">
        <RecipeDisplay
          meal={selectedMeal}
          onBackToMeals={handleBackToMeals}
          loadingGrocery={loadingGrocery}
        />
      </div>
    );
  }

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🍽️ Meal Suggestions</h3>
        <p>{data.summary}</p>
      </div>

      <div className="meals-grid">
        {meals.map((meal, index) => (
          <div key={index} className="meal-card" onClick={() => handleMealClick(meal)}>
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
                View Recipe & Get Prices
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Component for reverse meal search results
const ReverseMealResults = ({ data, onMealSelect }) => {
  const { suggested_meals, available_ingredients } = data;
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [showRecipe, setShowRecipe] = useState(false);
  const [loadingGrocery, setLoadingGrocery] = useState(false);

  const handleMealClick = async (meal) => {
    setSelectedMeal(meal);
    setShowRecipe(true);
    setLoadingGrocery(true);
    
    // Immediately start grocery analysis
    await onMealSelect(meal);
    setLoadingGrocery(false);
    
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
    setLoadingGrocery(false);
  };

  if (showRecipe && selectedMeal) {
    return (
      <div id="recipe-display" className="recipe-container">
        <RecipeDisplay
          meal={selectedMeal}
          onBackToMeals={handleBackToMeals}
          loadingGrocery={loadingGrocery}
        />
      </div>
    );
  }

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🥗 Meals You Can Make</h3>
        <p>Using your ingredients: {available_ingredients.join(', ')}</p>
      </div>

      <div className="meals-grid">
        {suggested_meals.map((meal, index) => (
          <div key={index} className="meal-card" onClick={() => handleMealClick(meal)}>
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

export default MealResults;
export { ReverseMealResults };