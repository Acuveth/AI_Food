import React, { useState } from 'react';
import ApiService from '../services/api';

const ProductSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sortBy, setSortBy] = useState('price');

  const popularSearches = [
    '🥛 Mleko', '🍞 Kruh', '🥚 Jajca', '🧀 Sir', 
    '🍎 Jabolka', '🥔 Krompir', '🍝 Testenine', '☕ Kava'
  ];

  const searchProducts = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await ApiService.searchProducts(searchQuery);
      
      if (response.success) {
        let results = response.data.products || [];
        
        // Sort results
        if (sortBy === 'price') {
          results.sort((a, b) => (a.current_price || 0) - (b.current_price || 0));
        } else if (sortBy === 'store') {
          results.sort((a, b) => (a.store_name || '').localeCompare(b.store_name || ''));
        }
        
        setSearchResults(results);
      } else {
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Error searching products:', error);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      searchProducts();
    }
  };

  const handlePopularSearch = (term) => {
    setSearchQuery(term.split(' ')[1]); // Remove emoji
    setSearchQuery(term.split(' ')[1]);
    setTimeout(() => searchProducts(), 100);
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
  };

  const formatPrice = (price) => {
    return `€${(price || 0).toFixed(2)}`;
  };

  const getStoreColor = (store) => {
    const colors = {
      'dm': '#1e3a8a',
      'lidl': '#0ea5e9',
      'mercator': '#dc2626',
      'spar': '#16a34a',
      'tus': '#7c3aed'
    };
    return colors[store?.toLowerCase()] || '#6b7280';
  };

  return (
    <div className="search-container">
      <div className="search-header">
        <h2>🔍 Product Search</h2>
        <p>Find the best prices across Slovenia's major grocery stores</p>
      </div>

      <div className="search-input-container">
        <div className="search-input">
          <div className="input-wrapper">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Search for any product... (e.g., milk, bread, coffee)"
              disabled={loading}
            />
            {searchQuery && (
              <button className="clear-button" onClick={clearSearch}>
                ✕
              </button>
            )}
          </div>
          <button 
            className="search-button"
            onClick={searchProducts} 
            disabled={loading || !searchQuery.trim()}
          >
            <span className="search-icon">
              {loading ? '⏳' : '🔍'}
            </span>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {searchResults.length > 0 && (
          <div className="search-controls">
            <div className="sort-controls">
              <label>Sort by:</label>
              <select 
                value={sortBy} 
                onChange={(e) => setSortBy(e.target.value)}
                className="sort-select"
              >
                <option value="price">💰 Price (Low to High)</option>
                <option value="store">🏪 Store Name</option>
              </select>
            </div>
            <div className="results-count">
              Found {searchResults.length} product{searchResults.length !== 1 ? 's' : ''}
            </div>
          </div>
        )}
      </div>

      {!searchQuery && !loading && searchResults.length === 0 && (
        <div className="search-welcome">
          <div className="search-hero">
            <h3>🛒 Start Your Smart Shopping Journey</h3>
            <p>Compare prices instantly across DM, Lidl, Mercator, SPAR, and TUS</p>
          </div>
          
          <div className="popular-searches">
            <h4>Popular searches:</h4>
            <div className="popular-grid">
              {popularSearches.map((term, index) => (
                <button
                  key={index}
                  className="popular-search-btn"
                  onClick={() => handlePopularSearch(term)}
                >
                  {term}
                </button>
              ))}
            </div>
          </div>

          <div className="search-tips">
            <h4>💡 Search Tips:</h4>
            <ul>
              <li>Try searching in Slovenian (e.g., "mleko" instead of "milk")</li>
              <li>Use general terms for better results (e.g., "kruh" not "črni kruh")</li>
              <li>Search for brands or specific product names</li>
            </ul>
          </div>
        </div>
      )}
      
      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <h3>Searching across stores...</h3>
            <p>Finding the best prices for you</p>
          </div>
        </div>
      )}
      
      <div className="search-results">
        {searchResults.length === 0 && !loading && searchQuery && (
          <div className="no-results">
            <div className="no-results-icon">😔</div>
            <h3>No products found</h3>
            <p>We couldn't find any products matching "{searchQuery}"</p>
            <div className="no-results-suggestions">
              <h4>Try:</h4>
              <ul>
                <li>Checking your spelling</li>
                <li>Using more general terms</li>
                <li>Searching in Slovenian</li>
              </ul>
            </div>
          </div>
        )}
        
        {searchResults.map((product, index) => (
          <div key={index} className="product-card enhanced">
            <div className="product-header">
              <h3>{product.product_name}</h3>
              {product.has_discount && (
                <div className="discount-badge">
                  🔥 {product.discount_percentage}% OFF
                </div>
              )}
            </div>
            
            <div className="product-pricing">
              <div className="current-price">{formatPrice(product.current_price)}</div>
              {product.regular_price && product.has_discount && (
                <div className="original-price">{formatPrice(product.regular_price)}</div>
              )}
            </div>

            <div className="product-store">
              <div 
                className="store-badge enhanced"
                style={{ backgroundColor: getStoreColor(product.store_name) }}
              >
                🏪 {product.store_name?.toUpperCase()}
              </div>
            </div>

            {product.ai_health_score && (
              <div className="health-score">
                <span className="health-label">Health Score:</span>
                <div className="health-bar">
                  <div 
                    className="health-fill"
                    style={{ 
                      width: `${(product.ai_health_score / 10) * 100}%`,
                      backgroundColor: product.ai_health_score >= 7 ? '#10b981' : 
                                     product.ai_health_score >= 5 ? '#f59e0b' : '#ef4444'
                    }}
                  ></div>
                </div>
                <span className="health-value">{product.ai_health_score}/10</span>
              </div>
            )}

            {product.ai_main_category && (
              <div className="product-category">
                📂 {product.ai_main_category}
              </div>
            )}

            {product.savings && product.has_discount && (
              <div className="savings-amount">
                💰 Save {formatPrice(product.savings)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProductSearch;