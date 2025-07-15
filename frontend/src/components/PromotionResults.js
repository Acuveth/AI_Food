// Updated Results Components with Slovenian Language Support

// PromotionResults.js - Slovenian Support
import React, { useState } from 'react';


const PromotionResults = ({ data }) => {
  const { promotions, analysis, category_breakdown, store_breakdown } = data;

  const formatStoreDisplay = (storeName) => {
    const storeDisplayNames = {
      'dm': 'DM',
      'lidl': 'Lidl',
      'mercator': 'Mercator',
      'spar': 'SPAR',
      'tus': 'Tuš',
      'hofer': 'Hofer'
    };
    return storeDisplayNames[storeName.toLowerCase()] || storeName.toUpperCase();
  };

  const formatDealQuality = (quality) => {
    const qualityTranslations = {
      'excellent': 'odličen',
      'good': 'dober',
      'fair': 'povprečen',
      'modest': 'skromen'
    };
    return qualityTranslations[quality] || quality;
  };

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🏷️ Najdeni akcijski izdelki</h3>
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
            <div className="store-info">{formatStoreDisplay(promo.store_name)}</div>
            <div className="discount-badge">
              {promo.discount_percentage}% POPUST
            </div>
            {promo.deal_quality && (
              <div className={`deal-quality ${promo.deal_quality}`}>
                {formatDealQuality(promo.deal_quality)} posel
              </div>
            )}
            {promo.savings_amount && (
              <div className="savings-info">
                Prihranek: €{promo.savings_amount.toFixed(2)}
              </div>
            )}
          </div>
        ))}
      </div>

      {analysis?.highlights && (
        <div className="analysis-summary">
          <h4>📊 Poudarki analize</h4>
          <div className="highlights-grid">
            <div className="highlight-card">
              <strong>Največji popust</strong>
              <p>{analysis.highlights.best_discount.product} v {formatStoreDisplay(analysis.highlights.best_discount.store)}</p>
              <span className="highlight-value">{analysis.highlights.best_discount.discount}% POPUST</span>
            </div>
            <div className="highlight-card">
              <strong>Največji prihranek</strong>
              <p>{analysis.highlights.biggest_savings.product}</p>
              <span className="highlight-value">€{analysis.highlights.biggest_savings.savings}</span>
            </div>
            {analysis.highlights.best_value && (
              <div className="highlight-card">
                <strong>Najboljše razmerje</strong>
                <p>{analysis.highlights.best_value.product}</p>
                <span className="highlight-value">Ocena: {analysis.highlights.best_value.value_score}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {category_breakdown && category_breakdown.length > 0 && (
        <div className="category-analysis">
          <h4>📈 Analiza po kategorijah</h4>
          <div className="category-grid">
            {category_breakdown.slice(0, 6).map((cat, index) => (
              <div key={index} className="category-card">
                <h5>{cat.category}</h5>
                <p>{cat.count} izdelkov</p>
                <p>Povprečni popust: {cat.avg_discount}%</p>
                <p>Skupni prihranek: €{cat.total_savings?.toFixed(2)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {store_breakdown && store_breakdown.length > 0 && (
        <div className="store-analysis">
          <h4>🏪 Analiza po trgovinah</h4>
          <div className="store-comparison">
            {store_breakdown.map((store, index) => (
              <div key={index} className="store-summary">
                <h5>{formatStoreDisplay(store.store)}</h5>
                <div className="store-stats">
                  <span>{store.count} akcij</span>
                  <span>Povprečen popust: {store.avg_discount}%</span>
                  <span>Kategorije: {store.categories.length}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// PriceComparisonResults.js - Slovenian Support
const PriceComparisonResults = ({ data }) => {
  const { results_by_store, price_analysis, best_deals, store_rankings } = data;

  const formatStoreDisplay = (storeName) => {
    const storeDisplayNames = {
      'dm': 'DM',
      'lidl': 'Lidl',
      'mercator': 'Mercator',
      'spar': 'SPAR',
      'tus': 'Tuš',
      'hofer': 'Hofer'
    };
    return storeDisplayNames[storeName.toLowerCase()] || storeName.toUpperCase();
  };

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🔍 Rezultati primerjave cen</h3>
        <p>{data.summary}</p>
      </div>

      <div className="price-comparison-grid">
        {Object.entries(results_by_store).map(([store, storeData]) => (
          <div key={store} className={`store-card ${storeData.product_count === 0 ? 'no-products' : ''}`}>
            <h4>{formatStoreDisplay(storeData.store_name)}</h4>
            
            {storeData.product_count > 0 ? (
              <>
                <div className="store-stats">
                  <p><strong>{storeData.product_count}</strong> najdenih izdelkov</p>
                  <p>Povprečje: <strong>€{storeData.avg_price?.toFixed(2)}</strong></p>
                  {storeData.cheapest_product && (
                    <p>Najceneje: <strong>€{storeData.cheapest_product.current_price?.toFixed(2)}</strong></p>
                  )}
                </div>
                
                <div className="products-list">
                  {storeData.products.slice(0, 3).map((product, index) => (
                    <div key={index} className="product-item">
                      <span className="product-name">{product.product_name}</span>
                      <span className="product-price">€{product.current_price?.toFixed(2)}</span>
                      {product.has_discount && (
                        <span className="discount-indicator">-{product.discount_percentage}%</span>
                      )}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="no-products-message">V tej trgovini ni najdenih izdelkov</p>
            )}
          </div>
        ))}
      </div>

      {price_analysis?.cheapest_option && (
        <div className="best-deal-highlight">
          <h4>🏆 Najbolje ponudba</h4>
          <div className="best-deal-card">
            <strong>{price_analysis.cheapest_option.product_name}</strong>
            <p>v {formatStoreDisplay(price_analysis.cheapest_option.store)}</p>
            <span className="best-price">€{price_analysis.cheapest_option.price?.toFixed(2)}</span>
            {price_analysis.cheapest_option.has_discount && (
              <div className="discount-info">
                Popust: {price_analysis.cheapest_option.discount_percentage}%
              </div>
            )}
          </div>
        </div>
      )}

      {price_analysis?.savings_potential && (
        <div className="savings-analysis">
          <h4>💰 Analiza prihrankov</h4>
          <div className="savings-info">
            <p>Maksimalni prihranek: <strong>€{price_analysis.savings_potential.max_savings?.toFixed(2)}</strong></p>
            <p>Razpon cen: <strong>{price_analysis.savings_potential.savings_percentage?.toFixed(1)}%</strong></p>
            <p className="recommendation">{price_analysis.savings_potential.recommendation}</p>
          </div>
        </div>
      )}

      {store_rankings && store_rankings.length > 0 && (
        <div className="store-rankings">
          <h4>🥇 Lestvica trgovin</h4>
          <div className="rankings-list">
            {store_rankings.slice(0, 5).map((ranking, index) => (
              <div key={index} className="ranking-item">
                <span className="rank-number">{ranking.rank}</span>
                <span className="store-name">{formatStoreDisplay(ranking.store)}</span>
                <span className="rank-score">Ocena: {ranking.rank_score}</span>
                {ranking.cheapest_price && (
                  <span className="cheapest-price">€{ranking.cheapest_price.toFixed(2)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// GroceryAnalysisResults.js - Slovenian Support
const GroceryAnalysisResults = ({ data }) => {
  const { meal, grocery_analysis } = data;
  const { store_analysis, combined_analysis, meal_statistics } = grocery_analysis;

  const formatStoreDisplay = (storeName) => {
    const storeDisplayNames = {
      'dm': 'DM',
      'lidl': 'Lidl',
      'mercator': 'Mercator',
      'spar': 'SPAR',
      'tus': 'Tuš',
      'hofer': 'Hofer'
    };
    return storeDisplayNames[storeName.toLowerCase()] || storeName.toUpperCase();
  };

  return (
    <div className="results-container">
      <div className="message-content">
        <h3>🛒 Analiza stroškov nakupovanja: {meal.title}</h3>
        <p>{data.summary}</p>
      </div>

      <div className="grocery-analysis">
        <div className="cost-summary">
          <div className="total-cost-card">
            <h4>Skupni ocenjeni strošek</h4>
            <span className="total-amount">€{combined_analysis.total_cost?.toFixed(2)}</span>
            <p>Strošek na porcijo: €{meal_statistics.cost_per_serving?.toFixed(2)}</p>
            <p>Najdene sestavine: {meal_statistics.ingredients_found}/{meal_statistics.total_ingredients}</p>
          </div>
        </div>

        <div className="store-comparison">
          <h4>Primerjava po trgovinah</h4>
          <div className="stores-grid">
            {Object.entries(store_analysis).map(([store, analysis]) => (
              <div key={store} className={`store-cost-card ${analysis.completeness === 100 ? 'best-option' : ''}`}>
                <h5>{formatStoreDisplay(analysis.store_name)}</h5>
                <div className="cost-info">
                  <span className="store-total">€{analysis.total_cost?.toFixed(2)}</span>
                  <span className="completeness">{analysis.completeness?.toFixed(0)}% kompletno</span>
                </div>
                <p>{analysis.available_items}/{grocery_analysis.ingredient_results ? Object.keys(grocery_analysis.ingredient_results).length : 0} sestavin</p>
                {analysis.missing_items?.length > 0 && (
                  <div className="missing-items">
                    <small>Manjka: {analysis.missing_items.slice(0, 2).join(', ')}</small>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="ingredients-breakdown">
          <h4>Pregled cen sestavin</h4>
          <div className="ingredients-list">
            {combined_analysis.item_details?.map((item, index) => (
              <div key={index} className="ingredient-row">
                <div className="ingredient-info">
                  {item.found && item.product?.product_name ? (
                    <>
                      <span className="ingredient-name">{item.product.product_name}</span>
                      <span className="ingredient-search-term">
                        za: {item.ingredient}
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="ingredient-name">{item.ingredient}</span>
                      <span className="ingredient-not-found-note">
                        Ni najden ustrezen izdelek
                      </span>
                    </>
                  )}
                </div>
                <div className="ingredient-price">
                  {item.found ? (
                    <>
                      <span className="price">€{item.price?.toFixed(2)}</span>
                      <span className="store">{formatStoreDisplay(item.store)}</span>
                    </>
                  ) : (
                    <span className="not-found">Ni najdeno</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="shopping-tips">
          <h4>💡 Nasveti za nakupovanje</h4>
          <div className="tips-grid">
            <div className="tip-card">
              <h5>Najboljša trgovina</h5>
              <p>
                {Object.entries(store_analysis)
                  .sort((a, b) => b[1].completeness - a[1].completeness)[0]?.[1]?.store_name 
                  ? formatStoreDisplay(Object.entries(store_analysis).sort((a, b) => b[1].completeness - a[1].completeness)[0][1].store_name)
                  : 'Ni podatkov'}
              </p>
            </div>
            <div className="tip-card">
              <h5>Prihranek</h5>
              <p>
                {combined_analysis.completeness > 80 
                  ? 'Večino sestavin lahko kupite v eni trgovini' 
                  : 'Morda boste potrebovali obisk več trgovin'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// MealResults.js - Slovenian Support (Updated portion)
const MealResults = ({ data, onMealSelect }) => {
  const { meals, request_analysis } = data;
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [showRecipe, setShowRecipe] = useState(false);
  const [loadingGrocery, setLoadingGrocery] = useState(false);

  const formatDietaryLabels = (labels) => {
    const translations = {
      'vegetarian': 'vegetarijansko',
      'vegan': 'veganski',
      'gluten-free': 'brez glutena',
      'dairy-free': 'brez mleka',
      'healthy': 'zdravo',
      'low-fat': 'z malo maščobe',
      'organic': 'bio'
    };
    return labels.map(label => translations[label.toLowerCase()] || label);
  };

  const formatCuisineType = (cuisine) => {
    const translations = {
      'italian': 'italijanska',
      'chinese': 'kitajska',
      'mexican': 'mehiška',
      'indian': 'indijska',
      'american': 'ameriška',
      'thai': 'tajska',
      'greek': 'grška',
      'french': 'francoska',
      'japanese': 'japonska',
      'mediterranean': 'mediteranska'
    };
    return translations[cuisine.toLowerCase()] || cuisine;
  };

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
        <h3>🍽️ Predlogi jedi</h3>
        <p>{data.summary}</p>
      </div>

      {request_analysis?.dietary_restrictions && request_analysis.dietary_restrictions.length > 0 && (
        <div className="dietary-info-summary">
          <h4>🥗 Prehranske zahteve</h4>
          <div className="dietary-tags">
            {request_analysis.dietary_restrictions.map((restriction, index) => (
              <span key={index} className="dietary-tag">
                {restriction === 'vegetarian' ? 'vegetarijansko' : 
                 restriction === 'vegan' ? 'veganski' : 
                 restriction === 'gluten-free' ? 'brez glutena' : 
                 restriction}
              </span>
            ))}
          </div>
        </div>
      )}

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
                <span>👥 {meal.servings || 2} porcij</span>
                {meal.cuisine_type && (
                  <span>🌍 {formatCuisineType(meal.cuisine_type)}</span>
                )}
              </div>
              
              {meal.diet_labels?.length > 0 && (
                <div className="diet-labels">
                  {formatDietaryLabels(meal.diet_labels).slice(0, 2).map((diet, i) => (
                    <span key={i} className="diet-label">{diet}</span>
                  ))}
                </div>
              )}
              
              <button className="select-meal-btn">
                Poglej recept in cene
              </button>
            </div>
          </div>
        ))}
      </div>

      {meals.length === 0 && (
        <div className="no-results">
          <h4>😔 Ni najdenih jedi</h4>
          <p>Poskusite z drugačnimi iskalnimi pojmi ali sprostite prehranske omejitve.</p>
        </div>
      )}
    </div>
  );
};

// RecipeDisplay.js - Slovenian Support (Updated portion)
const RecipeDisplay = ({ meal, onBackToMeals, loadingGrocery }) => {
  if (!meal) return null;

  const formatDietaryLabels = (labels) => {
    const translations = {
      'vegetarian': 'vegetarijansko',
      'vegan': 'veganski',
      'gluten-free': 'brez glutena',
      'dairy-free': 'brez mleka',
      'healthy': 'zdravo',
      'low-fat': 'z malo maščobe',
      'organic': 'bio'
    };
    return labels.map(label => translations[label.toLowerCase()] || label);
  };

  const formatCuisineType = (cuisine) => {
    const translations = {
      'italian': 'italijanska',
      'chinese': 'kitajska',
      'mexican': 'mehiška',
      'indian': 'indijska',
      'american': 'ameriška',
      'thai': 'tajska',
      'greek': 'grška',
      'french': 'francoska',
      'japanese': 'japonska',
      'mediterranean': 'mediteranska'
    };
    return translations[cuisine.toLowerCase()] || cuisine;
  };

  const formatDifficulty = (difficulty) => {
    const translations = {
      'easy': 'enostavno',
      'medium': 'srednje',
      'hard': 'težko',
      'beginner': 'za začetnike',
      'intermediate': 'za napredne',
      'advanced': 'za strokovnjake'
    };
    return translations[difficulty.toLowerCase()] || difficulty;
  };

  return (
    <div className="recipe-display-container">
      <div className="recipe-header">
        <button className="back-to-meals-btn" onClick={onBackToMeals}>
          ← Nazaj na možnosti jedi
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
            <h3>📋 Pregled recepta</h3>
            <div className="overview-grid">
              <div className="overview-item">
                <span className="overview-label">Čas priprave:</span>
                <span>{meal.prep_time || 0} minut</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Čas kuhanja:</span>
                <span>{meal.cook_time || 0} minut</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Skupni čas:</span>
                <span>{(meal.prep_time || 0) + (meal.cook_time || 0)} minut</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Število porcij:</span>
                <span>{meal.servings || 2}</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Težavnost:</span>
                <span className="difficulty-badge">{formatDifficulty(meal.difficulty || 'Medium')}</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Kuhinja:</span>
                <span>{formatCuisineType(meal.cuisine_type || 'International')}</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="recipe-right">
          {/* Ingredients List */}
          <div className="ingredients-section">
            <h3>🛒 Sestavine</h3>
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
              <p className="no-ingredients">Seznam sestavin ni na voljo za ta recept.</p>
            )}
          </div>
          
          {/* Diet Labels & Allergen Info */}
          {(meal.diet_labels?.length > 0 || meal.allergen_info?.length > 0) && (
            <div className="dietary-info">
              {meal.diet_labels?.length > 0 && (
                <div className="diet-labels">
                  <h4>🥗 Prehranske informacije</h4>
                  <div className="labels-list">
                    {formatDietaryLabels(meal.diet_labels).map((label, index) => (
                      <span key={index} className="diet-label">{label}</span>
                    ))}
                  </div>
                </div>
              )}
              
              {meal.allergen_info?.length > 0 && (
                <div className="allergen-info">
                  <h4>⚠️ Informacije o alergenih</h4>
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
            <h3>👨‍🍳 Navodila za pripravo</h3>
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
                <p>Podrobna navodila niso na voljo za ta recept.</p>
                {meal.recipe_url && (
                  <p>
                    <a 
                      href={meal.recipe_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="external-recipe-link"
                    >
                      Ogled celotnega recepta na izvirni strani →
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
                🔗 Ogled izvirnega recepta
              </a>
            )}
            
            <div className="grocery-status">
              {loadingGrocery ? (
                <div className="grocery-loading-status">
                  <div className="mini-spinner"></div>
                  <span>🛒 Iščem cene živil...</span>
                </div>
              ) : (
                <div className="grocery-ready-status">
                  <span>✅ Cene živil se bodo prikazale spodaj</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default { PromotionResults, PriceComparisonResults, GroceryAnalysisResults, MealResults, RecipeDisplay };