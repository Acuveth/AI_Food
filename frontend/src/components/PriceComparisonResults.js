// components/PriceComparisonResults.js - Fixed Export
import React from 'react';

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

export default PriceComparisonResults;