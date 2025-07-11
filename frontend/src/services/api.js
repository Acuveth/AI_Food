// services/api.js - Updated with meal details endpoint
const API_BASE_URL = 'http://localhost:8000';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  async sendChatMessage(message) {
    return this.request('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    });
  }

  async searchProducts(productName) {
    return this.request('/api/search', {
      method: 'POST',
      body: JSON.stringify({ product_name: productName }),
    });
  }

  // NEW: Get detailed meal information with grocery integration
  async getMealDetails(mealId, mealData) {
    return this.request(`/api/meals/details/${mealId}`, {
      method: 'POST',
      body: JSON.stringify({ meal_data: mealData }),
    });
  }

  // NEW: Search meals (returns cards without grocery integration)
  async searchMeals(request, maxResults = 8) {
    return this.request('/api/meals/search', {
      method: 'POST',
      body: JSON.stringify({ 
        request: request,
        max_results: maxResults,
        include_grocery: false  // Don't include grocery by default
      }),
    });
  }

  async healthCheck() {
    return this.request('/api/health');
  }
}

export default new ApiService();