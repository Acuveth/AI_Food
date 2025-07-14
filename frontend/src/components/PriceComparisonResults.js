// components/PriceComparisonResults.js
import React from 'react';

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

export default PriceComparisonResults;