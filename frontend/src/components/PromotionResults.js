// components/PromotionResults.js
import React from 'react';

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

export default PromotionResults;