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
        setSearchResults(response.data.products);
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

  return (
    <div className="search-container">
      <div className="search-input">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search for products..."
          disabled={loading}
        />
        <button onClick={searchProducts} disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      {loading && <div className="loading">Searching products...</div>}
      
      <div className="search-results">
        {searchResults.length === 0 && !loading && searchQuery && (
          <div className="no-results">
            <p>No products found for "{searchQuery}"</p>
          </div>
        )}
        
        {searchResults.map((product, index) => (
          <div key={index} className="product-card">
            <h3>{product.name}</h3>
            <p className="price">€{product.price.toFixed(2)}</p>
            <p className="store">{product.store}</p>
            <p className="location">{product.location}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProductSearch;