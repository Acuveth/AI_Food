import React, { useState } from 'react';
import ApiService from '../services/api';

const ProductSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const searchProducts = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await ApiService.searchProducts(searchQuery);
      
      if (response.success) {
        let results = response.data.products || [];
        results.sort((a, b) => (a.current_price || 0) - (b.current_price || 0));
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

  const formatPrice = (price) => {
    return `€${(price || 0).toFixed(2)}`;
  };

  return (
    <div className="search-container">
      <div className="search-header">
        <h2>Product Search</h2>
        <p>Find products across Slovenia's grocery stores</p>
      </div>

      <div className="search-input">
        <div className="input-wrapper">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search for products... (e.g. milk, bread)"
            disabled={loading}
          />
        </div>
        <button 
          className="search-button"
          onClick={searchProducts} 
          disabled={loading || !searchQuery.trim()}
        >
          <span>{loading ? '⏳' : '🔍'}</span>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <h3>Searching...</h3>
            <p>Finding products for you</p>
          </div>
        </div>
      )}
      
      <div className="search-results">
        {searchResults.length === 0 && !loading && searchQuery && (
          <div className="no-results">
            <h3>No products found</h3>
            <p>Try searching with different terms</p>
          </div>
        )}
        
        {searchResults.map((product, index) => (
          <div key={index} className="product-card">
            <h3>{product.product_name}</h3>
            <div className="price">{formatPrice(product.current_price)}</div>
            <div className="store">{product.store_name?.toUpperCase()}</div>
            {product.has_discount && (
              <div className="discount">🔥 {product.discount_percentage}% OFF</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProductSearch;