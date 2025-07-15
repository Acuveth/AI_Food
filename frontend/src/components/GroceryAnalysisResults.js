// components/GroceryAnalysisResults.js - Fixed
import React from 'react';

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
                <div className="ingredient-info">
                  {item.found && item.product?.product_name ? (
                    <>
                      <span className="ingredient-name">{item.product.product_name}</span>
                      <span className="ingredient-search-term">
                        for: {item.ingredient}
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="ingredient-name">{item.ingredient}</span>
                      <span className="ingredient-not-found-note">
                        No matching product found
                      </span>
                    </>
                  )}
                </div>
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

export default GroceryAnalysisResults;