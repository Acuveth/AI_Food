#!/usr/bin/env python3
"""
Input Interpretation Module - SLOVENIAN SUPPORT
Analyzes user input to understand their intent and route to appropriate modules
Updated with Slovenian keywords, patterns, and LLM prompts
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
    Enhanced with Slovenian language support
    """
    
    def __init__(self):
        self.client = client
        
        # Intent detection prompt - UPDATED FOR SLOVENIAN
        self.intent_detection_prompt = """
        You are an AI assistant that interprets user requests for a Slovenian grocery shopping app.
        Users may write in SLOVENIAN or ENGLISH. The database contains Slovenian product names.
        
        Analyze this user input and determine their intent: "{user_input}"
        
        Available intents:
        1. FIND_PROMOTIONS - User wants to find discounted/promotional items (akcije, popusti, ugodne ponudbe)
        2. COMPARE_ITEM_PRICES - User wants to compare prices of a specific item across stores (primerjaj cene, kje najceneje)
        3. SEARCH_MEALS - User wants to find meal/recipe suggestions (recepti, jedi, kuhanje)
        4. REVERSE_MEAL_SEARCH - User has ingredients and wants to know what meals they can make (imam sestavine, kaj lahko skuham)
        5. GENERAL_QUESTION - General grocery/food related question (splošno vprašanje)
        6. UNCLEAR - Intent is not clear (nejasno)
        
        Examples:
        - "Find deals on milk" / "Najdi akcije za mleko" → FIND_PROMOTIONS
        - "Where can I buy bread cheapest?" / "Kje najceneje kupim kruh?" → COMPARE_ITEM_PRICES
        - "I want Italian dinner recipes" / "Želim italijanske recepte za večerjo" → SEARCH_MEALS
        - "I have chicken, rice, and vegetables, what can I cook?" / "Imam piščanca, riž in zelenjavo, kaj lahko skuham?" → REVERSE_MEAL_SEARCH
        - "What's the healthiest breakfast?" / "Kaj je najzdravejši zajtrk?" → GENERAL_QUESTION
        
        IMPORTANT: Extract both English and Slovenian product names when present.
        
        Respond with JSON:
        {{
            "intent": "INTENT_NAME",
            "confidence": 0.0-1.0,
            "language_detected": "slovenian/english/mixed",
            "extracted_entities": {{
                "items": ["item1", "artikel1"],
                "cuisine_type": "cuisine",
                "meal_type": "breakfast/zajtrk/lunch/kosilo/dinner/večerja",
                "dietary_requirements": ["vegetarian/vegetarijansko", "vegan/veganski"],
                "store_preference": "dm/lidl/mercator/spar/tus",
                "price_sensitivity": "budget/poceni/expensive/drago/any",
                "ingredients": ["ingredient1", "sestavina1"]
            }},
            "reasoning": "Why this intent was chosen",
            "suggested_parameters": {{
                "search_term": "extracted search term in slovenian",
                "max_results": 10,
                "additional_filters": {{}}
            }}
        }}
        """
        
        # Enhanced entity extraction patterns - SLOVENIAN + ENGLISH
        self.entity_patterns = {
            "promotion_keywords": [
                # English
                "deals", "discount", "promotion", "sale", "cheap", "offer", "bargain", "special",
                # Slovenian
                "akcija", "akcije", "popust", "popusti", "ugodno", "poceni", "znižano", "znižanje",
                "posebna ponudba", "ugodna cena", "razprodaja", "cenovno ugodno"
            ],
            "comparison_keywords": [
                # English
                "compare", "cheapest", "price", "where", "cost", "expensive", "best price",
                # Slovenian
                "primerjaj", "najcenejši", "najceneje", "cena", "cene", "kje", "drago", "stroški",
                "najboljša cena", "kje kupiti", "kje dobim", "najugodneje"
            ],
            "meal_keywords": [
                # English
                "recipe", "meal", "cook", "dinner", "lunch", "breakfast", "food", "dish",
                # Slovenian
                "recept", "recepti", "jed", "jedi", "kuhaj", "kuhanje", "večerja", "kosilo", 
                "zajtrk", "hrana", "kuhinja", "pripravi", "skuhaj"
            ],
            "stores": ["dm", "lidl", "mercator", "spar", "tuš", "tus"],
            "meal_types": [
                # English
                "breakfast", "lunch", "dinner", "snack", "dessert",
                # Slovenian
                "zajtrk", "kosilo", "večerja", "prigrizek", "sladica", "malica"
            ],
            "cuisines": [
                # English
                "italian", "chinese", "mexican", "indian", "slovenian", "mediterranean", "asian",
                # Slovenian
                "italijanska", "kitajska", "mehiška", "indijska", "slovenska", "mediteranska", "azijska"
            ],
            "diets": [
                # English
                "vegetarian", "vegan", "gluten-free", "keto", "healthy", "organic",
                # Slovenian
                "vegetarijansko", "veganski", "brez glutena", "keto", "zdravo", "bio", "organski"
            ],
            "common_foods_si": [
                # Slovenian food items for better recognition
                "mleko", "kruh", "sir", "meso", "piščanec", "goveje", "svinjina", "riba",
                "riž", "testenine", "krompir", "čebula", "paradižnik", "jajca", "maslo",
                "jogurt", "kava", "čaj", "pivo", "vino", "voda", "sok", "sadje", "zelenjava"
            ]
        }
    
    async def interpret_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to interpret user input and determine intent
        Enhanced for Slovenian language support
        """
        logger.info(f"🧠 Interpreting user input: '{user_input}'")
        
        try:
            # Use LLM for intent detection with Slovenian support
            llm_result = await self._llm_intent_detection(user_input)
            
            # Validate and enhance with pattern matching
            enhanced_result = await self._enhance_with_patterns(user_input, llm_result)
            
            # Add routing information
            enhanced_result["routing"] = self._get_routing_info(enhanced_result["intent"])
            
            logger.info(f"✅ Detected intent: {enhanced_result['intent']} (confidence: {enhanced_result['confidence']})")
            logger.info(f"🌍 Language detected: {enhanced_result.get('language_detected', 'unknown')}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Input interpretation failed: {e}")
            return self._get_fallback_interpretation(user_input)
    
    async def _llm_intent_detection(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to detect user intent with Slovenian support"""
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
        """Enhanced pattern matching with Slovenian support"""
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
            # English
            "chicken", "beef", "pork", "fish", "rice", "pasta", "bread", "milk", 
            "eggs", "cheese", "tomato", "onion", "garlic", "potato", "carrot",
            # Slovenian
            "piščanec", "piščanca", "goveje", "govej", "svinjina", "riba", "riž", 
            "testenine", "kruh", "mleko", "jajca", "sir", "paradižnik", "čebula", 
            "česen", "krompir", "korenje", "zelenjava", "meso"
        ]
        
        for ingredient in common_ingredients:
            if ingredient in user_input_lower:
                ingredients_mentioned.append(ingredient)
        
        # Adjust intent based on patterns
        if len(ingredients_mentioned) >= 2 and any(word in user_input_lower for word in ["cook", "make", "kuhaj", "skuham", "pripravim"]):
            llm_result["intent"] = "REVERSE_MEAL_SEARCH"
            llm_result["extracted_entities"]["ingredients"] = ingredients_mentioned
            llm_result["confidence"] = min(llm_result.get("confidence", 0.7) + 0.2, 1.0)
        
        # Extract additional entities with Slovenian support
        entities = llm_result.get("extracted_entities", {})
        
        # Extract store mentions
        for store in self.entity_patterns["stores"]:
            if store in user_input_lower:
                entities["store_preference"] = store
        
        # Extract meal type (Slovenian + English)
        for meal_type in self.entity_patterns["meal_types"]:
            if meal_type in user_input_lower:
                entities["meal_type"] = meal_type
        
        # Extract cuisine (Slovenian + English)
        for cuisine in self.entity_patterns["cuisines"]:
            if cuisine in user_input_lower:
                entities["cuisine_type"] = cuisine
        
        # Extract dietary requirements (Slovenian + English)
        mentioned_diets = []
        for diet in self.entity_patterns["diets"]:
            if diet in user_input_lower:
                mentioned_diets.append(diet)
        if mentioned_diets:
            entities["dietary_requirements"] = mentioned_diets
        
        # Extract Slovenian food items
        mentioned_foods = []
        for food in self.entity_patterns["common_foods_si"]:
            if food in user_input_lower:
                mentioned_foods.append(food)
        if mentioned_foods:
            entities["items"] = entities.get("items", []) + mentioned_foods
        
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
        """Provide fallback interpretation when LLM fails - with Slovenian support"""
        user_input_lower = user_input.lower()
        
        # Simple keyword-based fallback with Slovenian support
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
            "language_detected": "unknown",
            "extracted_entities": {
                "items": [],
                "search_term": user_input
            },
            "reasoning": "Fallback interpretation using keyword matching (English + Slovenian)",
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
        """Generate clarification questions for unclear input - in Slovenian"""
        clarification_prompt = f"""
        The user said: "{unclear_input}"
        
        This input is unclear. Generate 3-4 helpful clarification questions IN SLOVENIAN that would help understand what the user wants to do in a grocery shopping context.
        
        Focus on:
        1. What specific product/item they're looking for (Kateri specifični izdelek iščejo)
        2. What action they want to take (Katero dejanje želijo izvesti - najti akcije, primerjati cene, dobiti recepte)
        3. Any specific preferences (Specifične preference - trgovina, cenovni razpon, prehranske potrebe)
        
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
                "Kateri specifični izdelek iščete?",
                "Želite najti akcije, primerjati cene ali dobiti predloge za recepte?", 
                "Imate preference glede trgovine (DM, Lidl, Mercator, SPAR, Tuš)?"
            ]

# Global interpreter instance
interpreter = InputInterpreter()

async def interpret_user_input(user_input: str) -> Dict[str, Any]:
    """Main function to interpret user input with Slovenian support"""
    return await interpreter.interpret_input(user_input)