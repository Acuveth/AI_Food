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
        <h1>Grocery Intelligence</h1>
        <p>Find the best grocery prices in Slovenia</p>
      </header>
      
      <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="main-content">
        {renderActiveComponent()}
      </main>

      <footer className="app-footer">
        <p>© 2024 Grocery Intelligence</p>
      </footer>
    </div>
  );
}

export default App;