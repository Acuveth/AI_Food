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

  async sendChatMessage(message, model = 'gpt-3.5-turbo') {
    return this.request('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message, model }),
    });
  }

  async searchProducts(productName, location = 'Ljubljana') {
    return this.request('/api/search', {
      method: 'POST',
      body: JSON.stringify({ product_name: productName, location }),
    });
  }

  async getPromotions() {
    return this.request('/api/promotions');
  }

  async healthCheck() {
    return this.request('/api/health');
  }
}

export default new ApiService();