import React, { useState, useEffect } from 'react';
import TabNavigation from './components/TabNavigation';
import ChatAssistant from './components/ChatAssistant';
import ProductSearch from './components/ProductSearch';
import Promotions from './components/Promotions';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Add loading animation
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatAssistant />;
      case 'search':
        return <ProductSearch />;
      case 'promotions':
        return <Promotions />;
      default:
        return <ChatAssistant />;
    }
  };

  if (!isLoaded) {
    return (
      <div className="App">
        <div className="loading">
          <h2>Loading Slovenian Grocery Intelligence...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      
      <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="main-content">
        {renderActiveComponent()}
      </main>

      <footer className="app-footer">
        <p>© 2024 Slovenian Grocery Intelligence | Powered by AI</p>
      </footer>
    </div>
  );
}

export default App;