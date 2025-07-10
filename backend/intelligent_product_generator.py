#!/usr/bin/env python3
"""
Intelligent Product Generator - Think First, Then Search
Generates comprehensive product lists before searching the database
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class IntelligentProductGenerator:
    """
    AI-powered product generator that thinks first, then searches
    """
    
    def __init__(self):
        self.client = client
        
    async def generate_health_focused_products(self, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate health-focused product list first, then search database"""
        
        prompt = f"""
        You are a nutritionist helping someone find the healthiest grocery products.
        
        Generate a comprehensive list of 15-20 healthy products that should be available in Slovenian grocery stores.
        Focus on products with high nutritional value, low processing, and health benefits.
        
        Consider these categories:
        - Fresh vegetables and fruits
        - Whole grains and legumes
        - Lean proteins
        - Healthy fats
        - Dairy or dairy alternatives
        - Minimally processed foods
        
        For each product, provide:
        1. Product name in Slovenian (primary search term)
        2. Alternative names/variants
        3. Why it's healthy
        4. Expected health score (1-10)
        
        Format as JSON:
        {{
            "products": [
                {{
                    "name": "mleko brez laktoze",
                    "alternatives": ["lactose free milk", "mleko laktoze"],
                    "health_reason": "Good protein, easier digestion",
                    "expected_health_score": 7
                }}
            ],
            "search_strategy": "comprehensive health search",
            "total_products": 15
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            generated_products = json.loads(json_text)
            
            logger.info(f"🧠 Generated {len(generated_products.get('products', []))} health-focused products")
            return generated_products
            
        except Exception as e:
            logger.error(f"Error generating health products: {e}")
            return {"products": [], "search_strategy": "fallback", "total_products": 0}
    
    async def generate_diet_compatible_products(self, diet_type: str) -> Dict[str, Any]:
        """Generate diet-specific product list first, then search database"""
        
        prompt = f"""
        You are a diet specialist helping someone find products for a {diet_type} diet.
        
        Generate a comprehensive list of 12-15 products that are perfect for a {diet_type} diet and should be available in Slovenian grocery stores (DM, Lidl, Mercator, SPAR, TUS).
        
        For {diet_type} diet, focus on:
        - Products that strictly comply with {diet_type} requirements
        - Common ingredients for {diet_type} meals
        - Protein sources suitable for {diet_type}
        - Snacks and convenience foods that are {diet_type}
        - Cooking ingredients and condiments
        
        For each product, provide:
        1. Product name in Slovenian (primary search term)
        2. Alternative names/variants
        3. Why it fits {diet_type} diet
        4. Category (protein, carbs, fats, etc.)
        
        Format as JSON:
        {{
            "products": [
                {{
                    "name": "mandljevo mleko",
                    "alternatives": ["almond milk", "rastlinsko mleko"],
                    "diet_reason": "Plant-based, dairy-free",
                    "category": "dairy_alternative"
                }}
            ],
            "diet_type": "{diet_type}",
            "search_strategy": "diet-specific search",
            "total_products": 12
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            generated_products = json.loads(json_text)
            
            logger.info(f"🥗 Generated {len(generated_products.get('products', []))} {diet_type} products")
            return generated_products
            
        except Exception as e:
            logger.error(f"Error generating {diet_type} products: {e}")
            return {"products": [], "diet_type": diet_type, "search_strategy": "fallback", "total_products": 0}
    
    async def generate_meal_planning_products(self, meal_type: str, people_count: int = 1, budget: float = None) -> Dict[str, Any]:
        """Generate meal planning product list first, then search database"""
        
        budget_text = f"with a budget of €{budget}" if budget else "with flexible budget"
        
        prompt = f"""
        You are a meal planning expert helping someone create a {meal_type} for {people_count} people {budget_text}.
        
        Generate a comprehensive shopping list of 10-15 products needed for a complete {meal_type} for {people_count} people.
        
        Consider:
        - Base ingredients for {meal_type}
        - Protein sources
        - Vegetables and sides
        - Cooking essentials (oil, spices, etc.)
        - Portions for {people_count} people
        - Slovenian grocery store availability
        
        For each product, provide:
        1. Product name in Slovenian (primary search term)
        2. Alternative names
        3. Quantity needed for {people_count} people
        4. Role in the meal (protein, vegetable, etc.)
        5. Priority (essential, nice-to-have)
        
        Format as JSON:
        {{
            "products": [
                {{
                    "name": "piščančji file",
                    "alternatives": ["chicken breast", "piščanec"],
                    "quantity": "500g",
                    "role": "protein",
                    "priority": "essential"
                }}
            ],
            "meal_type": "{meal_type}",
            "people_count": {people_count},
            "budget": {budget},
            "search_strategy": "meal planning search",
            "total_products": 12
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            generated_products = json.loads(json_text)
            
            logger.info(f"🍽️ Generated {len(generated_products.get('products', []))} products for {meal_type} planning")
            return generated_products
            
        except Exception as e:
            logger.error(f"Error generating meal planning products: {e}")
            return {"products": [], "meal_type": meal_type, "search_strategy": "fallback", "total_products": 0}
    
    async def generate_seasonal_products(self, season: str = None) -> Dict[str, Any]:
        """Generate seasonal product list first, then search database"""
        
        import datetime
        if not season:
            month = datetime.datetime.now().month
            if month in [12, 1, 2]:
                season = "winter"
            elif month in [3, 4, 5]:
                season = "spring"
            elif month in [6, 7, 8]:
                season = "summer"
            else:
                season = "autumn"
        
        prompt = f"""
        You are a seasonal food expert helping someone find the best {season} products.
        
        Generate a list of 12-15 products that are in season, fresh, and at their best during {season} in Slovenia.
        
        Consider:
        - Fresh fruits and vegetables in season
        - Seasonal specialties
        - Products that taste best in {season}
        - Traditional {season} foods
        - Storage and preservation items for {season}
        
        For each product, provide:
        1. Product name in Slovenian (primary search term)
        2. Alternative names
        3. Why it's perfect for {season}
        4. Freshness indicators to look for
        5. Category (fruit, vegetable, specialty, etc.)
        
        Format as JSON:
        {{
            "products": [
                {{
                    "name": "jabolka",
                    "alternatives": ["apple", "jabolko"],
                    "season_reason": "Harvest season, crisp and fresh",
                    "freshness_tips": "Look for firm, unblemished skin",
                    "category": "fruit"
                }}
            ],
            "season": "{season}",
            "search_strategy": "seasonal search",
            "total_products": 12
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            generated_products = json.loads(json_text)
            
            logger.info(f"🌿 Generated {len(generated_products.get('products', []))} seasonal products for {season}")
            return generated_products
            
        except Exception as e:
            logger.error(f"Error generating seasonal products: {e}")
            return {"products": [], "season": season, "search_strategy": "fallback", "total_products": 0}

    async def generate_smart_shopping_products(self, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate smart shopping product list focusing on deals and value"""
        
        prompt = f"""
        You are a smart shopping expert helping someone find the best deals and value products.
        
        Generate a list of 15-20 products that typically have good deals, bulk discounts, or represent excellent value in Slovenian grocery stores.
        
        Focus on:
        - Staple foods that are frequently on sale
        - Products with long shelf life good for bulk buying
        - Store brands vs name brands opportunities
        - Seasonal sale items
        - Multi-buy deals products
        
        For each product, provide:
        1. Product name in Slovenian (primary search term)
        2. Alternative names/brands
        3. Why it's a smart buy
        4. Best buying strategy
        5. Category
        
        Format as JSON:
        {{
            "products": [
                {{
                    "name": "testenine",
                    "alternatives": ["pasta", "špageti"],
                    "smart_reason": "Long shelf life, frequently on sale",
                    "buying_strategy": "Buy in bulk when discounted",
                    "category": "staple"
                }}
            ],
            "search_strategy": "smart shopping search",
            "total_products": 15
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            generated_products = json.loads(json_text)
            
            logger.info(f"🛒 Generated {len(generated_products.get('products', []))} smart shopping products")
            return generated_products
            
        except Exception as e:
            logger.error(f"Error generating smart shopping products: {e}")
            return {"products": [], "search_strategy": "fallback", "total_products": 0}

    async def generate_allergen_safe_products(self, avoid_allergens: List[str]) -> Dict[str, Any]:
        """Generate allergen-safe product list first, then search database"""
        
        allergen_text = ", ".join(avoid_allergens)
        
        prompt = f"""
        You are an allergen specialist helping someone avoid {allergen_text}.
        
        Generate a comprehensive list of 12-15 products that are safe for someone avoiding {allergen_text} and should be available in Slovenian grocery stores.
        
        Focus on:
        - Products naturally free from {allergen_text}
        - Certified allergen-free alternatives
        - Safe protein sources
        - Safe grains and starches
        - Safe dairy alternatives if applicable
        - Safe snacks and convenience foods
        
        For each product, provide:
        1. Product name in Slovenian (primary search term)
        2. Alternative names
        3. Why it's safe from {allergen_text}
        4. Category (protein, grain, dairy, etc.)
        5. Safety confidence level (high, medium)
        
        Format as JSON:
        {{
            "products": [
                {{
                    "name": "riževi kosmiči",
                    "alternatives": ["rice flakes", "rice cereal"],
                    "safety_reason": "Naturally gluten-free grain",
                    "category": "grain",
                    "safety_confidence": "high"
                }}
            ],
            "avoided_allergens": ["{allergen_text}"],
            "search_strategy": "allergen-safe search",
            "total_products": 12
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            generated_products = json.loads(json_text)
            
            logger.info(f"🛡️ Generated {len(generated_products.get('products', []))} allergen-safe products")
            return generated_products
            
        except Exception as e:
            logger.error(f"Error generating allergen-safe products: {e}")
            return {"products": [], "avoided_allergens": avoid_allergens, "search_strategy": "fallback", "total_products": 0}

if __name__ == "__main__":
    # Test the generator
    async def test_generator():
        generator = IntelligentProductGenerator()
        result = await generator.generate_health_focused_products()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    asyncio.run(test_generator())