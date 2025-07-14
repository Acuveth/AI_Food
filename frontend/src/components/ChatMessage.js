// components/ChatMessage.js
import React from 'react';
import PromotionResults from './PromotionResults';
import PriceComparisonResults from './PriceComparisonResults';
import MealResults, { ReverseMealResults } from './MealResults';
import GroceryAnalysisResults from './GroceryAnalysisResults';
import GeneralHelpResults, { ClarificationResults } from './HelpResults';
import RecipeDisplay from './RecipeDisplay';

const ChatMessage = ({ msg, onMealSelect, onSuggestionClick }) => {
  const formatTime = (timestamp) => {
    return timestamp ? timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  };

  const renderMessageContent = () => {
    const { intent, approach, data } = msg;

    // Handle different types of responses
    if (intent === 'FIND_PROMOTIONS' && data?.promotions) {
      return <PromotionResults data={data} />;
    }
    
    if (intent === 'COMPARE_ITEM_PRICES' && data?.results_by_store) {
      return <PriceComparisonResults data={data} />;
    }
    
    if (intent === 'SEARCH_MEALS' && data?.meals) {
      return <MealResults data={data} onMealSelect={onMealSelect} />;
    }
    
    if (intent === 'REVERSE_MEAL_SEARCH' && data?.suggested_meals) {
      return <ReverseMealResults data={data} onMealSelect={onMealSelect} />;
    }
    
    if (approach === 'meal_grocery_analysis' && data?.grocery_analysis) {
      return <GroceryAnalysisResults data={data} />;
    }
    
    if (intent === 'GENERAL_QUESTION' && data?.suggestions) {
      return <GeneralHelpResults data={data} onSuggestionClick={onSuggestionClick} />;
    }
    
    if (approach === 'clarification_needed' && data?.clarification_questions) {
      return <ClarificationResults data={data} onSuggestionClick={onSuggestionClick} />;
    }

    // Default text content
    return (
      <div className="message-content">
        {msg.content}
      </div>
    );
  };

  return (
    <div className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
      <div className="message-avatar">
        {msg.role === 'user' ? '👤' : '🛒'}
      </div>
      <div className="message-bubble">
        {renderMessageContent()}
        {msg.timestamp && (
          <div className="message-time">
            {formatTime(msg.timestamp)}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;