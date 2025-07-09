import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';

const Promotions = () => {
  const [promotions, setPromotions] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchPromotions = async () => {
    setLoading(true);
    try {
      const response = await ApiService.getPromotions();
      if (response.success) {
        setPromotions(response.data.promotions);
      }
    } catch (error) {
      console.error('Error fetching promotions:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPromotions();
  }, []);

  return (
    <div className="promotions-container">
      <div className="promotions-header">
        <h2>Current Promotions</h2>
        <button onClick={fetchPromotions} disabled={loading}>
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>
      
      {loading && (
        <div className="loading">Loading promotions...</div>
      )}
      
      <div className="promotions-grid">
        {promotions.length === 0 && !loading && (
          <div className="no-promotions">
            <p>No promotions available at the moment.</p>
          </div>
        )}
        
        {promotions.map((promotion, index) => (
          <div key={index} className="promotion-card">
            <h3>{promotion.product}</h3>
            <div className="store-badge">{promotion.store}</div>
            <div className="price-info">
              <span className="original-price">€{promotion.original_price.toFixed(2)}</span>
              <span className="discount-price">€{promotion.discount_price.toFixed(2)}</span>
            </div>
            <div className="discount-percent">
              {promotion.discount_percent}% OFF
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Promotions;