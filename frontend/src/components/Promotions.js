import React, { useState, useEffect } from 'react';

const Promotions = () => {
  const [promotions, setPromotions] = useState([]);
  const [loading, setLoading] = useState(false);

  const mockPromotions = [
    {
      product_name: "Milk UHT 1L",
      store_name: "lidl",
      current_price: 0.89,
      regular_price: 1.19,
      discount_percentage: 25,
      has_discount: true
    },
    {
      product_name: "Bread 500g",
      store_name: "mercator", 
      current_price: 1.49,
      regular_price: 1.89,
      discount_percentage: 21,
      has_discount: true
    },
    {
      product_name: "Apples 1kg",
      store_name: "spar",
      current_price: 1.99,
      regular_price: 2.49,
      discount_percentage: 20,
      has_discount: true
    },
    {
      product_name: "Cooking Oil 1L",
      store_name: "tus",
      current_price: 2.29,
      regular_price: 2.99,
      discount_percentage: 23,
      has_discount: true
    },
    {
      product_name: "Yogurt 500g",
      store_name: "dm",
      current_price: 1.19,
      regular_price: 1.59,
      discount_percentage: 25,
      has_discount: true
    }
  ];

  const fetchPromotions = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setPromotions(mockPromotions);
      setLoading(false);
    }, 1000);
  };

  useEffect(() => {
    fetchPromotions();
  }, []);

  const formatPrice = (price) => `€${(price || 0).toFixed(2)}`;

  return (
    <div className="promotions-container">
      <div className="promotions-header">
        <h2>Current Promotions</h2>
        <p>Today's best deals across Slovenia</p>
      </div>
      
      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <h3>Loading promotions...</h3>
            <p>Finding the best deals</p>
          </div>
        </div>
      )}
      
      <div className="promotions-grid">
        {promotions.map((promotion, index) => (
          <div key={index} className="promotion-card">
            <h3>{promotion.product_name}</h3>
            <div className="price">{formatPrice(promotion.current_price)}</div>
            <div className="store">{promotion.store_name?.toUpperCase()}</div>
            {promotion.has_discount && (
              <div className="discount">
                🔥 {promotion.discount_percentage}% OFF - Save {formatPrice(promotion.regular_price - promotion.current_price)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Promotions;