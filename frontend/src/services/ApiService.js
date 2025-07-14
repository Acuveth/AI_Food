// services/ApiService.js - Updated for streamlined backend
class ApiService {
  static BASE_URL = 'http://localhost:8000';

  static async request(endpoint, options = {}) {
    const url = `${this.BASE_URL}${endpoint}`;
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

  // Main intelligent request endpoint
  static async sendIntelligentRequest(input) {
    return this.request('/api/intelligent-request', {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
  }

  // Direct function calls
  static async getPromotions(searchFilter = null) {
    const endpoint = searchFilter 
      ? `/api/promotions/all?search=${encodeURIComponent(searchFilter)}`
      : '/api/promotions/all';
    return this.request(endpoint);
  }

  static async comparePrices(itemName) {
    return this.request(`/api/compare-prices/${encodeURIComponent(itemName)}`);
  }

  static async searchMeals(request) {
    return this.request('/api/search-meals', {
      method: 'POST',
      body: JSON.stringify({ request }),
    });
  }

  static async analyzeMealGrocery(mealData) {
    return this.request('/api/meal-grocery-analysis', {
      method: 'POST',
      body: JSON.stringify({ meal_data: mealData }),
    });
  }

  static async findMealsFromIngredients(ingredients) {
    return this.request('/api/meals-from-ingredients', {
      method: 'POST',
      body: JSON.stringify({ ingredients }),
    });
  }

  static async healthCheck() {
    return this.request('/api/health');
  }
}

export default ApiService;