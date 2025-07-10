import React, { useState, useEffect } from 'react';

const DatabaseExplorer = () => {
  const [activeView, setActiveView] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState({});
  const [customQuery, setCustomQuery] = useState('');
  const [queryResults, setQueryResults] = useState([]);

  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    if (activeView === 'overview') {
      loadOverviewData();
    }
  }, [activeView]);

  const loadOverviewData = async () => {
    setLoading(true);
    try {
      // Load multiple data sources in parallel
      const [storesRes, categoriesRes, schemaRes] = await Promise.all([
        fetch(`${API_BASE}/api/database/stores`),
        fetch(`${API_BASE}/api/database/categories`),
        fetch(`${API_BASE}/api/database/schema`)
      ]);

      const [stores, categories, schema] = await Promise.all([
        storesRes.json(),
        categoriesRes.json(),
        schemaRes.json()
      ]);

      setData({ stores: stores.data, categories: categories.data, schema: schema.data });
    } catch (error) {
      console.error('Error loading overview data:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeCustomQuery = async () => {
    if (!customQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/database/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: customQuery })
      });
      
      const result = await response.json();
      if (result.success) {
        setQueryResults(result.data.results);
      } else {
        alert(`Query Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error executing query:', error);
      alert('Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const searchProducts = async (searchTerm, filters = {}) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/database/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ search_term: searchTerm, ...filters })
      });
      
      const result = await response.json();
      if (result.success) {
        setData({ ...data, searchResults: result.data.products });
      }
    } catch (error) {
      console.error('Error searching products:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => `€${(price || 0).toFixed(2)}`;

  const renderOverview = () => (
    <div className="database-overview">
      <h3>📊 Database Overview</h3>
      
      {data.schema && (
        <div className="schema-info">
          <h4>Database Schema</h4>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">{data.schema.total_tables || 0}</div>
              <div className="stat-label">Total Tables</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{Object.keys(data.schema.tables || {}).length}</div>
              <div className="stat-label">Data Tables</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{Object.keys(data.schema.views || {}).length}</div>
              <div className="stat-label">Views</div>
            </div>
          </div>
        </div>
      )}

      {data.stores?.stores && (
        <div className="stores-info">
          <h4>🏪 Store Statistics</h4>
          <div className="stores-grid">
            {data.stores.stores.map((store, index) => (
              <div key={index} className="store-card">
                <h5>{store.store_name?.toUpperCase()}</h5>
                <div className="store-stats">
                  <p><strong>{store.product_count}</strong> products</p>
                  <p>Avg: {formatPrice(store.avg_price)}</p>
                  <p>Range: {formatPrice(store.min_price)} - {formatPrice(store.max_price)}</p>
                  <p><strong>{store.discounted_products}</strong> on sale</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.categories?.categories && (
        <div className="categories-info">
          <h4>📂 Top Categories</h4>
          <div className="categories-list">
            {data.categories.categories.slice(0, 10).map((category, index) => (
              <div key={index} className="category-item">
                <div className="category-name">{category.ai_main_category}</div>
                <div className="category-stats">
                  <span>{category.product_count} products</span>
                  <span>Avg: {formatPrice(category.avg_price)}</span>
                  {category.avg_health_score && (
                    <span>Health: {category.avg_health_score.toFixed(1)}/10</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderQueryBuilder = () => (
    <div className="query-builder">
      <h3>🔍 Custom Query Builder</h3>
      
      <div className="query-examples">
        <h4>Quick Examples:</h4>
        <div className="example-buttons">
          <button onClick={() => setCustomQuery("SELECT store_name, COUNT(*) as products FROM unified_products_view GROUP BY store_name ORDER BY products DESC")}>
            Store Product Counts
          </button>
          <button onClick={() => setCustomQuery("SELECT product_name, store_name, current_price FROM unified_products_view WHERE has_discount = 1 ORDER BY discount_percentage DESC LIMIT 10")}>
            Top Discounts
          </button>
          <button onClick={() => setCustomQuery("SELECT ai_main_category, AVG(ai_health_score) as avg_health FROM unified_products_view WHERE ai_health_score IS NOT NULL GROUP BY ai_main_category ORDER BY avg_health DESC")}>
            Category Health Scores
          </button>
        </div>
      </div>

      <div className="query-input">
        <textarea
          value={customQuery}
          onChange={(e) => setCustomQuery(e.target.value)}
          placeholder="Enter your SQL SELECT query here..."
          rows={6}
          className="query-textarea"
        />
        <button 
          onClick={executeCustomQuery} 
          disabled={loading || !customQuery.trim()}
          className="execute-button"
        >
          {loading ? '⏳ Executing...' : '🚀 Execute Query'}
        </button>
      </div>

      {queryResults.length > 0 && (
        <div className="query-results">
          <h4>Query Results ({queryResults.length} rows):</h4>
          <div className="results-table">
            <table>
              <thead>
                <tr>
                  {Object.keys(queryResults[0] || {}).map(key => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {queryResults.slice(0, 50).map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      <td key={i}>{String(value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {queryResults.length > 50 && (
              <p className="results-note">Showing first 50 results of {queryResults.length}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const renderAdvancedSearch = () => {
    const [searchFilters, setSearchFilters] = useState({
      search_term: '',
      store: '',
      category: '',
      min_price: '',
      max_price: '',
      has_discount: null,
      min_health_score: '',
      limit: 20
    });

    const handleSearch = () => {
      const filters = Object.fromEntries(
        Object.entries(searchFilters).filter(([_, value]) => value !== '' && value !== null)
      );
      searchProducts(filters.search_term, filters);
    };

    return (
      <div className="advanced-search">
        <h3>🔍 Advanced Product Search</h3>
        
        <div className="search-filters">
          <div className="filter-row">
            <input
              type="text"
              placeholder="Search term (e.g., mleko, kruh)"
              value={searchFilters.search_term}
              onChange={(e) => setSearchFilters({...searchFilters, search_term: e.target.value})}
            />
            <select
              value={searchFilters.store}
              onChange={(e) => setSearchFilters({...searchFilters, store: e.target.value})}
            >
              <option value="">All Stores</option>
              <option value="dm">DM</option>
              <option value="lidl">Lidl</option>
              <option value="mercator">Mercator</option>
              <option value="spar">SPAR</option>
              <option value="tus">TUS</option>
            </select>
          </div>
          
          <div className="filter-row">
            <input
              type="number"
              placeholder="Min Price (€)"
              value={searchFilters.min_price}
              onChange={(e) => setSearchFilters({...searchFilters, min_price: e.target.value})}
              step="0.01"
            />
            <input
              type="number"
              placeholder="Max Price (€)"
              value={searchFilters.max_price}
              onChange={(e) => setSearchFilters({...searchFilters, max_price: e.target.value})}
              step="0.01"
            />
            <select
              value={searchFilters.has_discount || ''}
              onChange={(e) => setSearchFilters({...searchFilters, has_discount: e.target.value === '' ? null : e.target.value === 'true'})}
            >
              <option value="">Any Discount</option>
              <option value="true">Only Discounted</option>
              <option value="false">No Discount</option>
            </select>
          </div>
          
          <div className="filter-row">
            <input
              type="number"
              placeholder="Min Health Score (0-10)"
              value={searchFilters.min_health_score}
              onChange={(e) => setSearchFilters({...searchFilters, min_health_score: e.target.value})}
              min="0"
              max="10"
              step="0.1"
            />
            <input
              type="number"
              placeholder="Max Results"
              value={searchFilters.limit}
              onChange={(e) => setSearchFilters({...searchFilters, limit: parseInt(e.target.value) || 20})}
              min="1"
              max="200"
            />
            <button onClick={handleSearch} disabled={loading} className="search-button">
              {loading ? '⏳ Searching...' : '🔍 Search'}
            </button>
          </div>
        </div>

        {data.searchResults && (
          <div className="search-results">
            <h4>Search Results ({data.searchResults.length} products):</h4>
            <div className="products-grid">
              {data.searchResults.map((product, index) => (
                <div key={index} className="product-card">
                  <h5>{product.product_name}</h5>
                  <div className="product-info">
                    <div className="price">{formatPrice(product.current_price)}</div>
                    <div className="store">{product.store_name?.toUpperCase()}</div>
                    {product.has_discount && (
                      <div className="discount">🔥 {product.discount_percentage}% OFF</div>
                    )}
                    {product.ai_health_score && (
                      <div className="health-score">Health: {product.ai_health_score}/10</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="database-explorer">
      <div className="explorer-header">
        <h2>🗄️ Database Explorer</h2>
        <p>Explore your grocery database directly</p>
      </div>

      <div className="explorer-nav">
        <button 
          className={activeView === 'overview' ? 'active' : ''}
          onClick={() => setActiveView('overview')}
        >
          📊 Overview
        </button>
        <button 
          className={activeView === 'search' ? 'active' : ''}
          onClick={() => setActiveView('search')}
        >
          🔍 Advanced Search
        </button>
        <button 
          className={activeView === 'query' ? 'active' : ''}
          onClick={() => setActiveView('query')}
        >
          📝 Custom Queries
        </button>
      </div>

      <div className="explorer-content">
        {loading && activeView === 'overview' && (
          <div className="loading">Loading database information...</div>
        )}
        
        {activeView === 'overview' && renderOverview()}
        {activeView === 'search' && renderAdvancedSearch()}
        {activeView === 'query' && renderQueryBuilder()}
      </div>

      <style jsx>{`
        .database-explorer {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .explorer-header {
          text-align: center;
          margin-bottom: 30px;
        }

        .explorer-nav {
          display: flex;
          gap: 10px;
          margin-bottom: 30px;
          justify-content: center;
        }

        .explorer-nav button {
          padding: 12px 20px;
          border: none;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.1);
          color: var(--text-primary);
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .explorer-nav button.active {
          background: var(--success-gradient);
          color: white;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 15px;
          margin: 20px 0;
        }

        .stat-card {
          background: rgba(255, 255, 255, 0.1);
          padding: 20px;
          border-radius: 12px;
          text-align: center;
        }

        .stat-number {
          font-size: 2em;
          font-weight: bold;
          color: var(--text-primary);
        }

        .stat-label {
          font-size: 0.9em;
          color: var(--text-secondary);
        }

        .stores-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 15px;
          margin: 20px 0;
        }

        .store-card {
          background: rgba(255, 255, 255, 0.15);
          padding: 20px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .store-card h5 {
          margin: 0 0 15px 0;
          color: var(--text-primary);
          font-size: 1.2em;
        }

        .store-stats p {
          margin: 5px 0;
          font-size: 0.9em;
        }

        .categories-list {
          display: grid;
          gap: 10px;
          margin: 20px 0;
        }

        .category-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
        }

        .category-stats {
          display: flex;
          gap: 15px;
          font-size: 0.9em;
          color: var(--text-secondary);
        }

        .query-textarea {
          width: 100%;
          padding: 15px;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.3);
          background: rgba(255, 255, 255, 0.9);
          font-family: 'Courier New', monospace;
          font-size: 14px;
          resize: vertical;
        }

        .execute-button {
          margin-top: 10px;
          padding: 12px 24px;
          background: var(--success-gradient);
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 600;
        }

        .execute-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .results-table {
          margin-top: 20px;
          overflow-x: auto;
        }

        .results-table table {
          width: 100%;
          border-collapse: collapse;
          background: rgba(255, 255, 255, 0.9);
          border-radius: 8px;
          overflow: hidden;
        }

        .results-table th,
        .results-table td {
          padding: 10px;
          text-align: left;
          border-bottom: 1px solid #ddd;
        }

        .results-table th {
          background: var(--success-gradient);
          color: white;
          font-weight: 600;
        }

        .search-filters {
          background: rgba(255, 255, 255, 0.1);
          padding: 20px;
          border-radius: 12px;
          margin-bottom: 20px;
        }

        .filter-row {
          display: flex;
          gap: 15px;
          margin-bottom: 15px;
        }

        .filter-row input,
        .filter-row select {
          flex: 1;
          padding: 10px;
          border-radius: 6px;
          border: 1px solid rgba(255, 255, 255, 0.3);
          background: rgba(255, 255, 255, 0.9);
        }

        .products-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 15px;
          margin-top: 20px;
        }

        .product-card {
          background: rgba(255, 255, 255, 0.9);
          padding: 15px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .product-card h5 {
          margin: 0 0 10px 0;
          font-size: 1em;
          color: var(--text-primary);
        }

        .product-info {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .price {
          font-size: 1.3em;
          font-weight: bold;
          color: var(--text-primary);
        }

        .store {
          font-size: 0.9em;
          color: var(--text-secondary);
        }

        .discount {
          font-size: 0.8em;
          color: #f59e0b;
          font-weight: 600;
        }

        .health-score {
          font-size: 0.8em;
          color: #10b981;
        }

        .example-buttons {
          display: flex;
          gap: 10px;
          margin: 15px 0;
          flex-wrap: wrap;
        }

        .example-buttons button {
          padding: 8px 15px;
          background: rgba(102, 126, 234, 0.1);
          border: 1px solid rgba(102, 126, 234, 0.2);
          border-radius: 20px;
          cursor: pointer;
          font-size: 0.9em;
          transition: all 0.3s ease;
        }

        .example-buttons button:hover {
          background: rgba(102, 126, 234, 0.2);
        }

        .loading {
          text-align: center;
          padding: 40px;
          color: var(--text-secondary);
        }
      `}</style>
    </div>
  );
};

export default DatabaseExplorer;