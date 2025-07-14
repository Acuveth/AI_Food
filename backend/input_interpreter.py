#!/usr/bin/env python3
"""
Input Interpretation Module
Analyzes user input to understand their intent and route to appropriate modules
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class UserIntent(Enum):
    """Possible user intents"""
    FIND_PROMOTIONS = "find_promotions"
    COMPARE_ITEM_PRICES = "compare_item_prices"
    SEARCH_MEALS = "search_meals"
    REVERSE_MEAL_SEARCH = "reverse_meal_search"
    GENERAL_QUESTION = "general_question"
    UNCLEAR = "unclear"

class InputInterpreter:
    """
    Interprets user input to understand what they want to do
    """
    
    def __init__(self):
        self.client = client
        
        # Intent detection prompt
        self.intent_detection_prompt = """
        You are an AI assistant that interprets user requests for a Slovenian grocery shopping app.
        
        Analyze this user input and determine their intent: "{user_input}"
        
        Available intents:
        1. FIND_PROMOTIONS - User wants to find discounted/promotional items
        2. COMPARE_ITEM_PRICES - User wants to compare prices of a specific item across stores
        3. SEARCH_MEALS - User wants to find meal/recipe suggestions
        4. REVERSE_MEAL_SEARCH - User has ingredients and wants to know what meals they can make
        5. GENERAL_QUESTION - General grocery/food related question
        6. UNCLEAR - Intent is not clear
        
        Examples:
        - "Find deals on milk" → FIND_PROMOTIONS
        - "Where can I buy bread cheapest?" → COMPARE_ITEM_PRICES
        - "I want Italian dinner recipes" → SEARCH_MEALS
        - "I have chicken, rice, and vegetables, what can I cook?" → REVERSE_MEAL_SEARCH
        - "What's the healthiest breakfast?" → GENERAL_QUESTION
        
        Respond with JSON:
        {{
            "intent": "INTENT_NAME",
            "confidence": 0.0-1.0,
            "extracted_entities": {{
                "items": ["item1", "item2"],
                "cuisine_type": "cuisine",
                "meal_type": "breakfast/lunch/dinner",
                "dietary_requirements": ["vegetarian", "gluten-free"],
                "store_preference": "store_name",
                "price_sensitivity": "budget/expensive/any",
                "ingredients": ["ingredient1", "ingredient2"]
            }},
            "reasoning": "Why this intent was chosen",
            "suggested_parameters": {{
                "search_term": "extracted search term",
                "max_results": 10,
                "additional_filters": {{}};
            }}
        }}
        """
        
        # Entity extraction patterns
        self.entity_patterns = {
            "promotion_keywords": [
                "deals", "discount", "promotion", "sale", "cheap", "offer", 
                "akcija", "popust", "ugodno", "poceni", "znižano"
            ],
            "comparison_keywords": [
                "compare", "cheapest", "price", "where", "cost", "expensive",
                "primerjaj", "najcenejši", "cena", "kje", "drago"
            ],
            "meal_keywords": [
                "recipe", "meal", "cook", "dinner", "lunch", "breakfast", 
                "recept", "jed", "kuhaj", "večerja", "kosilo", "zajtrk"
            ],
            "stores": ["dm", "lidl", "mercator", "spar", "tus"],
            "meal_types": ["breakfast", "lunch", "dinner", "snack", "dessert"],
            "cuisines": ["italian", "chinese", "mexican", "indian", "slovenian"],
            "diets": ["vegetarian", "vegan", "gluten-free", "keto", "healthy"]
        }
    
    async def interpret_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to interpret user input and determine intent
        """
        logger.info(f"🧠 Interpreting user input: '{user_input}'")
        
        try:
            # Use LLM for intent detection
            llm_result = await self._llm_intent_detection(user_input)
            
            # Validate and enhance with pattern matching
            enhanced_result = await self._enhance_with_patterns(user_input, llm_result)
            
            # Add routing information
            enhanced_result["routing"] = self._get_routing_info(enhanced_result["intent"])
            
            logger.info(f"✅ Detected intent: {enhanced_result['intent']} (confidence: {enhanced_result['confidence']})")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Input interpretation failed: {e}")
            return self._get_fallback_interpretation(user_input)
    
    async def _llm_intent_detection(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to detect user intent"""
        prompt = self.intent_detection_prompt.format(user_input=user_input)
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
            
            result = json.loads(json_text)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise
    
    async def _enhance_with_patterns(self, user_input: str, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM results with pattern matching"""
        user_input_lower = user_input.lower()
        
        # Check for specific patterns that might indicate different intents
        promotion_score = sum(1 for keyword in self.entity_patterns["promotion_keywords"] 
                            if keyword in user_input_lower)
        comparison_score = sum(1 for keyword in self.entity_patterns["comparison_keywords"] 
                             if keyword in user_input_lower)
        meal_score = sum(1 for keyword in self.entity_patterns["meal_keywords"] 
                        if keyword in user_input_lower)
        
        # Check if user mentions specific ingredients (reverse meal search indicator)
        ingredients_mentioned = []
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "rice", "pasta", "bread", "milk", 
            "eggs", "cheese", "tomato", "onion", "garlic", "potato", "carrot",
            "piščanec", "goveje", "svinjina", "riba", "riž", "testenine", "kruh", 
            "mleko", "jajca", "sir", "paradižnik", "čebula", "česen", "krompir"
        ]
        
        for ingredient in common_ingredients:
            if ingredient in user_input_lower:
                ingredients_mentioned.append(ingredient)
        
        # Adjust intent based on patterns
        if len(ingredients_mentioned) >= 2 and "cook" in user_input_lower or "make" in user_input_lower:
            llm_result["intent"] = "REVERSE_MEAL_SEARCH"
            llm_result["extracted_entities"]["ingredients"] = ingredients_mentioned
            llm_result["confidence"] = min(llm_result.get("confidence", 0.7) + 0.2, 1.0)
        
        # Extract additional entities
        entities = llm_result.get("extracted_entities", {})
        
        # Extract store mentions
        for store in self.entity_patterns["stores"]:
            if store in user_input_lower:
                entities["store_preference"] = store
        
        # Extract meal type
        for meal_type in self.entity_patterns["meal_types"]:
            if meal_type in user_input_lower:
                entities["meal_type"] = meal_type
        
        # Extract cuisine
        for cuisine in self.entity_patterns["cuisines"]:
            if cuisine in user_input_lower:
                entities["cuisine_type"] = cuisine
        
        # Extract dietary requirements
        mentioned_diets = []
        for diet in self.entity_patterns["diets"]:
            if diet in user_input_lower:
                mentioned_diets.append(diet)
        if mentioned_diets:
            entities["dietary_requirements"] = mentioned_diets
        
        llm_result["extracted_entities"] = entities
        return llm_result
    
    def _get_routing_info(self, intent: str) -> Dict[str, Any]:
        """Get routing information based on detected intent"""
        routing_map = {
            "FIND_PROMOTIONS": {
                "module": "promotion_finder",
                "function": "find_promotions",
                "endpoint": "/api/promotions"
            },
            "COMPARE_ITEM_PRICES": {
                "module": "item_finder", 
                "function": "compare_item_prices",
                "endpoint": "/api/compare-prices"
            },
            "SEARCH_MEALS": {
                "module": "meal_search",
                "function": "search_meals",
                "endpoint": "/api/search-meals"
            },
            "REVERSE_MEAL_SEARCH": {
                "module": "meal_search",
                "function": "reverse_meal_search", 
                "endpoint": "/api/meals-from-ingredients"
            },
            "GENERAL_QUESTION": {
                "module": "general_handler",
                "function": "handle_general_question",
                "endpoint": "/api/general"
            },
            "UNCLEAR": {
                "module": "clarification_handler",
                "function": "request_clarification",
                "endpoint": "/api/clarify"
            }
        }
        
        return routing_map.get(intent, routing_map["UNCLEAR"])
    
    def _get_fallback_interpretation(self, user_input: str) -> Dict[str, Any]:
        """Provide fallback interpretation when LLM fails"""
        user_input_lower = user_input.lower()
        
        # Simple keyword-based fallback
        if any(keyword in user_input_lower for keyword in self.entity_patterns["promotion_keywords"]):
            intent = "FIND_PROMOTIONS"
        elif any(keyword in user_input_lower for keyword in self.entity_patterns["comparison_keywords"]):
            intent = "COMPARE_ITEM_PRICES"
        elif any(keyword in user_input_lower for keyword in self.entity_patterns["meal_keywords"]):
            intent = "SEARCH_MEALS"
        else:
            intent = "GENERAL_QUESTION"
        
        return {
            "intent": intent,
            "confidence": 0.3,
            "extracted_entities": {
                "items": [],
                "search_term": user_input
            },
            "reasoning": "Fallback interpretation using keyword matching",
            "suggested_parameters": {
                "search_term": user_input,
                "max_results": 10
            },
            "routing": self._get_routing_info(intent),
            "fallback_used": True
        }
    
    def validate_intent_result(self, result: Dict[str, Any]) -> bool:
        """Validate that the intent detection result is properly formatted"""
        required_fields = ["intent", "confidence", "extracted_entities", "routing"]
        return all(field in result for field in required_fields)
    
    async def get_clarification_questions(self, unclear_input: str) -> List[str]:
        """Generate clarification questions for unclear input"""
        clarification_prompt = f"""
        The user said: "{unclear_input}"
        
        This input is unclear. Generate 3-4 helpful clarification questions that would help understand what the user wants to do in a grocery shopping context.
        
        Focus on:
        1. What specific product/item they're looking for
        2. What action they want to take (find deals, compare prices, get recipes)
        3. Any specific preferences (store, price range, dietary needs)
        
        Return as JSON array: ["question1", "question2", "question3"]
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": clarification_prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
            
            questions = json.loads(json_text)
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate clarification questions: {e}")
            return [
                "What specific product are you looking for?",
                "Would you like to find deals, compare prices, or get recipe suggestions?", 
                "Do you have any store preferences (DM, Lidl, Mercator, SPAR, TUS)?"
            ]

# Global interpreter instance
interpreter = InputInterpreter()

async def interpret_user_input(user_input: str) -> Dict[str, Any]:
    """Main function to interpret user input"""
    return await interpreter.interpret_input(user_input)