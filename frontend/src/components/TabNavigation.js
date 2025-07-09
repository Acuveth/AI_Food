import React from 'react';

const TabNavigation = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'chat', label: 'Chat Assistant', icon: '💬' },
    { id: 'search', label: 'Product Search', icon: '🔍' },
    { id: 'promotions', label: 'Promotions', icon: '🏷️' },
  ];

  return (
    <nav className="tab-nav">
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={activeTab === tab.id ? 'active' : ''}
          onClick={() => setActiveTab(tab.id)}
        >
          <span className="tab-icon">{tab.icon}</span>
          {tab.label}
        </button>
      ))}
    </nav>
  );
};

export default TabNavigation;