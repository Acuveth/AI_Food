// components/PromotionResults.js - Fixed Export
import React from 'react';

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

export default PromotionResults;