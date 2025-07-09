import React, { useState } from 'react';
import TabNavigation from './components/TabNavigation';
import ChatAssistant from './components/ChatAssistant';
import ProductSearch from './components/ProductSearch';
import Promotions from './components/Promotions';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

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

  return (
    <div className="App">
      <header className="app-header">
        <h1>🛒 Slovenian Grocery Intelligence</h1>
        <p>AI-powered grocery shopping assistant</p>
      </header>

      <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="main-content">
        {renderActiveComponent()}
      </main>
    </div>
  );
}

export default App;