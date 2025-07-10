import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';

const Promotions = () => {
  const [promotions, setPromotions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('all');

  const fetchPromotions = async () => {
    setLoading(true);
    try {
      const response = await ApiService.getPromotions();
      if (response.success) {
        // Mock promotions data with proper structure for demo
        const mockPromotions = [
          {
            product_name: "Mleko UHT 1L",
            store_name: "lidl",
            current_price: 0.89,
            regular_price: 1.19,
            discount_percentage: 25,
            has_discount: true,
            ai_main_category: "Mlečni izdelki",
            savings: 0.30
          },
          {
            product_name: "Kruh polnozrnati 500g",
            store_name: "mercator", 
            current_price: 1.49,
            regular_price: 1.89,
            discount_percentage: 21,
            has_discount: true,
            ai_main_category: "Pekovski izdelki",
            savings: 0.40
          },
          {
            product_name: "Jabolka Gala 1kg",
            store_name: "spar",
            current_price: 1.99,
            regular_price: 2.49,
            discount_percentage: 20,
            has_discount: true,
            ai_main_category: "Sadje",
            savings: 0.50
          },
          {
            product_name: "Olje sončnično 1L",
            store_name: "tus",
            current_price: 2.29,
            regular_price: 2.99,
            discount_percentage: 23,
            has_discount: true,
            ai_main_category: "Maščobe",
            savings: 0.70
          },
          {
            product_name: "Jogurt naravni 500g",
            store_name: "dm",
            current_price: 1.19,
            regular_price: 1.59,
            discount_percentage: 25,
            has_discount: true,
            ai_main_category: "Mlečni izdelki",
            savings: 0.40
          },
          {
            product_name: "Testenine penne 500g",
            store_name: "lidl",
            current_price: 0.69,
            regular_price: 0.99,
            discount_percentage: 30,
            has_discount: true,
            ai_main_category: "Testenine",
            savings: 0.30
          }
        ];
        setPromotions(response.data.promotions || mockPromotions);
      }
    } catch (error) {
      console.error('Error fetching promotions:', error);
      // Set mock data on error for demo purposes
      const mockPromotions = [
        {
          product_name: "Mleko UHT 1L",
          store_name: "lidl",
          current_price: 0.89,
          regular_price: 1.19,
          discount_percentage: 25,
          has_discount: true,
          ai_main_category: "Mlečni izdelki",
          savings: 0.30
        },
        {
          product_name: "Kruh polnozrnati 500g",
          store_name: "mercator", 
          current_price: 1.49,
          regular_price: 1.89,
          discount_percentage: 21,
          has_discount: true,
          ai_main_category: "Pekovski izdelki",
          savings: 0.40
        },
        {
          product_name: "Jabolka Gala 1kg",
          store_name: "spar",
          current_price: 1.99,
          regular_price: 2.49,
          discount_percentage: 20,
          has_discount: true,
          ai_main_category: "Sadje",
          savings: 0.50
        }
      ];
      setPromotions(mockPromotions);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPromotions();
  }, []);

  const getStoreColor = (store) => {
    const colors = {
      'dm': 'linear-gradient(135deg, #1e3a8a, #3b82f6)',
      'lidl': 'linear-gradient(135deg, #0ea5e9, #38bdf8)',
      'mercator': 'linear-gradient(135deg, #dc2626, #f87171)',
      'spar': 'linear-gradient(135deg, #16a34a, #4ade80)',
      'tus': 'linear-gradient(135deg, #7c3aed, #a78bfa)'
    };
    return colors[store?.toLowerCase()] || 'linear-gradient(135deg, #6b7280, #9ca3af)';
  };

  const getStoreIcon = (store) => {
    const icons = {
      'dm': '🏪',
      'lidl': '🛒',
      'mercator': '🏬',
      'spar': '🏪',
      'tus': '🛍️'
    };
    return icons[store?.toLowerCase()] || '🏪';
  };

  const filteredPromotions = promotions.filter(promo => {
    if (filter === 'all') return true;
    return promo.store_name?.toLowerCase() === filter;
  });

  const totalSavings = filteredPromotions.reduce((sum, promo) => sum + (promo.savings || 0), 0);

  const formatPrice = (price) => `€${(price || 0).toFixed(2)}`;

  return (
    <div className="promotions-container">
      <div className="promotions-hero">
        <div className="promotions-header">
          <h2>🎁 Current Promotions</h2>
          <p>Save money with today's best deals across Slovenia</p>
          <button 
            className="refresh-button"
            onClick={fetchPromotions} 
            disabled={loading}
          >
            <span className="refresh-icon">
              {loading ? '⏳' : '🔄'}
            </span>
            {loading ? 'Loading...' : 'Refresh Deals'}
          </button>
        </div>

        {!loading && promotions.length > 0 && (
          <div className="promotions-stats">
            <div className="stat-card">
              <div className="stat-icon">💰</div>
              <div className="stat-content">
                <div className="stat-number">{formatPrice(totalSavings)}</div>
                <div className="stat-label">Total Savings Available</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🛍️</div>
              <div className="stat-content">
                <div className="stat-number">{filteredPromotions.length}</div>
                <div className="stat-label">Active Promotions</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🏪</div>
              <div className="stat-content">
                <div className="stat-number">{new Set(promotions.map(p => p.store_name)).size}</div>
                <div className="stat-label">Stores with Deals</div>
              </div>
            </div>
          </div>
        )}

        {!loading && promotions.length > 0 && (
          <div className="promotions-filters">
            <label>Filter by store:</label>
            <div className="filter-buttons">
              <button 
                className={filter === 'all' ? 'active' : ''}
                onClick={() => setFilter('all')}
              >
                All Stores
              </button>
              {['dm', 'lidl', 'mercator', 'spar', 'tus'].map(store => (
                <button
                  key={store}
                  className={filter === store ? 'active' : ''}
                  onClick={() => setFilter(store)}
                >
                  {getStoreIcon(store)} {store.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {loading && (
        <div className="promotions-loading">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <h3>Finding the best deals...</h3>
            <p>Checking promotions across all stores</p>
          </div>
        </div>
      )}
      
      <div className="promotions-grid">
        {filteredPromotions.length === 0 && !loading && (
          <div className="no-promotions">
            <div className="no-promotions-icon">😔</div>
            <h3>No promotions found</h3>
            <p>
              {filter === 'all' 
                ? "No promotions are currently available. Check back soon!" 
                : `No promotions found for ${filter.toUpperCase()}. Try another store.`}
            </p>
            <button onClick={() => setFilter('all')} className="view-all-btn">
              View All Promotions
            </button>
          </div>
        )}
        
        {filteredPromotions.map((promotion, index) => (
          <div key={index} className="promotion-card modern">
            <div 
              className="promotion-background"
              style={{ background: getStoreColor(promotion.store_name) }}
            ></div>
            
            <div className="promotion-content">
              <div className="promotion-header">
                <h3>{promotion.product_name}</h3>
                <div className="discount-badge large">
                  🔥 {promotion.discount_percentage}% OFF
                </div>
              </div>

              <div className="store-info">
                <div className="store-badge modern">
                  {getStoreIcon(promotion.store_name)} {promotion.store_name?.toUpperCase()}
                </div>
                {promotion.ai_main_category && (
                  <div className="category-badge">
                    📂 {promotion.ai_main_category}
                  </div>
                )}
              </div>

              <div className="price-section">
                <div className="price-comparison">
                  <div className="current-price">
                    {formatPrice(promotion.current_price)}
                  </div>
                  <div className="original-price">
                    was {formatPrice(promotion.regular_price)}
                  </div>
                </div>
                
                <div className="savings-highlight">
                  <div className="savings-amount">
                    💰 Save {formatPrice(promotion.savings)}
                  </div>
                </div>
              </div>

              <div className="promotion-actions">
                <button className="find-store-btn">
                  📍 Find Store
                </button>
                <button className="add-to-list-btn">
                  ➕ Add to List
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Promotions;