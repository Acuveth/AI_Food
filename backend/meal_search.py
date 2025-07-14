#!/usr/bin/env python3
"""
Meal Search Module
Handles meal requests using external APIs and integrates with database for ingredient pricing
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import aiohttp
from openai import OpenAI
import os
from dotenv import load_dotenv
from database_handler import get_db_handler

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MealSearcher:
    """
    Handles meal search using external APIs and grocery database integration
    """
    
    def __init__(self):
        self.db_handler = None
        
        # API configurations
        self.apis = {
            "spoonacular": {
                "base_url": "https://api.spoonacular.com/recipes",
                "api_key": os.getenv("SPOONACULAR_API_KEY"),
                "enabled": bool(os.getenv("SPOONACULAR_API_KEY"))
            },
            "edamam": {
                "base_url": "https://api.edamam.com/api/recipes/v2",
                "app_id": os.getenv("EDAMAM_APP_ID"),
                "app_key": os.getenv("EDAMAM_APP_KEY"),
                "enabled": bool(os.getenv("EDAMAM_APP_ID") and os.getenv("EDAMAM_APP_KEY"))
            },
            "themealdb": {
                "base_url": "https://www.themealdb.com/api/json/v1/1",
                "enabled": True  # Free API
            }
        }
        
        logger.info(f"🔧 Meal APIs: Spoonacular={self.apis['spoonacular']['enabled']}, "
                   f"Edamam={self.apis['edamam']['enabled']}, TheMealDB={self.apis['themealdb']['enabled']}")
    
    async def _ensure_db_connection(self):
        """Ensure database connection is available"""
        if self.db_handler is None:
            self.db_handler = await get_db_handler()
    
    async def search_meals(
        self,
        user_request: str,
        max_results: int = 12,
        include_grocery_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Search for meals based on user request
        
        Args:
            user_request: User's meal request in natural language
            max_results: Maximum number of meals to return
            include_grocery_analysis: Whether to include grocery price analysis
        """
        logger.info(f"🍽️ Searching meals for: '{user_request}'")
        
        try:
            # Step 1: Interpret the meal request
            request_analysis = await self._analyze_meal_request(user_request)
            
            # Step 2: Search meals across APIs
            all_meals = []
            
            async with aiohttp.ClientSession() as session:
                # Search Spoonacular
                if self.apis["spoonacular"]["enabled"]:
                    spoon_meals = await self._search_spoonacular(session, request_analysis, max_results // 3)
                    all_meals.extend(spoon_meals)
                
                # Search Edamam
                if self.apis["edamam"]["enabled"]:
                    edamam_meals = await self._search_edamam(session, request_analysis, max_results // 3)
                    all_meals.extend(edamam_meals)
                
                # Search TheMealDB
                themealdb_meals = await self._search_themealdb(session, request_analysis, max_results // 3)
                all_meals.extend(themealdb_meals)
            
            # Step 3: Filter and rank meals
            filtered_meals = await self._filter_and_rank_meals(all_meals, request_analysis, max_results)
            
            # Step 4: Add basic nutritional estimates
            for meal in filtered_meals:
                meal["estimated_nutrition"] = self._estimate_nutrition(meal)
            
            result = {
                "success": True,
                "meals": filtered_meals,
                "total_found": len(all_meals),
                "filtered_count": len(filtered_meals),
                "request_analysis": request_analysis,
                "apis_used": [api for api, config in self.apis.items() if config["enabled"]],
                "summary": self._generate_meal_search_summary(user_request, filtered_meals, request_analysis)
            }
            
            logger.info(f"✅ Found {len(filtered_meals)} meals from {len(all_meals)} total results")
            return result
            
        except Exception as e:
            logger.error(f"❌ Meal search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "meals": [],
                "message": "Failed to search for meals"
            }
    
    async def get_meal_with_grocery_analysis(
        self,
        meal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed grocery analysis for a selected meal
        
        Args:
            meal_data: Complete meal data from the meal card
        """
        await self._ensure_db_connection()
        
        logger.info(f"🛒 Analyzing grocery prices for meal: {meal_data.get('title', 'Unknown')}")
        
        try:
            # Extract ingredients
            ingredients = meal_data.get("ingredients", [])
            if not ingredients:
                return {
                    "success": False,
                    "message": "No ingredients found in meal data",
                    "meal": meal_data
                }
            
            # Search for ingredients in grocery database
            ingredient_results = await self.db_handler.find_meal_ingredients(
                [ing.get("name", ing.get("original", "")) for ing in ingredients]
            )
            
            # Analyze store costs
            store_analysis = self._analyze_store_costs(ingredient_results)
            
            # Calculate combined cheapest option
            combined_analysis = self._analyze_combined_cheapest(ingredient_results)
            
            # Calculate meal statistics
            meal_stats = self._calculate_meal_statistics(meal_data, ingredient_results, combined_analysis)
            
            result = {
                "success": True,
                "meal": meal_data,
                "grocery_analysis": {
                    "ingredient_results": ingredient_results,
                    "store_analysis": store_analysis,
                    "combined_analysis": combined_analysis,
                    "meal_statistics": meal_stats,
                    "shopping_recommendations": self._generate_shopping_recommendations(store_analysis, combined_analysis)
                },
                "summary": self._generate_grocery_analysis_summary(meal_data, store_analysis, combined_analysis)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Grocery analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "meal": meal_data,
                "message": "Failed to analyze grocery prices"
            }
    
    async def reverse_meal_search(
        self,
        available_ingredients: List[str],
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Find meals that can be made with available ingredients
        
        Args:
            available_ingredients: List of ingredients user has available
            max_results: Maximum number of meal suggestions
        """
        logger.info(f"🔍 Reverse meal search with ingredients: {available_ingredients}")
        
        try:
            # Use LLM to understand ingredients and suggest meal types
            meal_suggestions = await self._suggest_meals_from_ingredients(available_ingredients)
            
            # Search for meals based on suggestions
            all_suggested_meals = []
            
            async with aiohttp.ClientSession() as session:
                for suggestion in meal_suggestions:
                    search_analysis = {
                        "meal_type": suggestion.get("meal_type", "any"),
                        "cuisine_type": suggestion.get("cuisine", "any"),
                        "search_keywords": suggestion.get("keywords", []),
                        "included_ingredients": available_ingredients
                    }
                    
                    # Search each API with ingredient focus
                    if self.apis["spoonacular"]["enabled"]:
                        meals = await self._search_spoonacular_by_ingredients(session, available_ingredients, max_results // 3)
                        all_suggested_meals.extend(meals)
                    
                    if self.apis["edamam"]["enabled"]:
                        meals = await self._search_edamam_by_ingredients(session, available_ingredients, max_results // 3)
                        all_suggested_meals.extend(meals)
            
            # Rank by ingredient match
            ranked_meals = self._rank_by_ingredient_match(all_suggested_meals, available_ingredients)
            
            # Limit results
            final_meals = ranked_meals[:max_results]
            
            return {
                "success": True,
                "available_ingredients": available_ingredients,
                "suggested_meals": final_meals,
                "total_found": len(all_suggested_meals),
                "ingredient_suggestions": meal_suggestions,
                "summary": f"Found {len(final_meals)} meals you can make with your available ingredients"
            }
            
        except Exception as e:
            logger.error(f"❌ Reverse meal search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "available_ingredients": available_ingredients,
                "message": "Failed to find meals with your ingredients"
            }
    
    # MEAL REQUEST ANALYSIS
    async def _analyze_meal_request(self, user_request: str) -> Dict[str, Any]:
        """Analyze user's meal request to extract parameters"""
        prompt = f"""
        Analyze this meal request: "{user_request}"
        
        Extract:
        1. Meal type (breakfast, lunch, dinner, snack, any)
        2. Cuisine preferences (italian, chinese, mexican, etc.)
        3. Dietary restrictions (vegetarian, vegan, gluten-free, keto, etc.)
        4. Cooking time preferences (quick <30min, medium 30-60min, elaborate >60min)
        5. Number of servings needed
        6. Specific ingredients mentioned
        7. Health preferences (healthy, comfort food, balanced)
        8. Difficulty level (easy, medium, hard)
        
        Respond with JSON:
        {{
            "meal_type": "dinner",
            "cuisine_types": ["italian"],
            "dietary_restrictions": ["vegetarian"],
            "max_cook_time": 60,
            "servings": 4,
            "included_ingredients": ["chicken", "rice"],
            "excluded_ingredients": ["nuts"],
            "health_focus": "healthy",
            "difficulty": "medium",
            "search_keywords": ["italian", "dinner", "healthy"],
            "english_query": "healthy Italian dinner recipes"
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=600
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
            
            analysis = json.loads(json_text)
            return analysis
            
        except Exception as e:
            logger.error(f"Meal request analysis failed: {e}")
            return {
                "meal_type": "any",
                "cuisine_types": [],
                "dietary_restrictions": [],
                "max_cook_time": 60,
                "servings": 2,
                "included_ingredients": [],
                "excluded_ingredients": [],
                "health_focus": "balanced",
                "difficulty": "medium",
                "search_keywords": [user_request],
                "english_query": user_request
            }
    
    # API SEARCH METHODS
    async def _search_spoonacular(self, session: aiohttp.ClientSession, analysis: Dict, max_results: int) -> List[Dict]:
        """Search Spoonacular API"""
        if not self.apis["spoonacular"]["enabled"]:
            return []
        
        try:
            params = {
                "apiKey": self.apis["spoonacular"]["api_key"],
                "number": max_results,
                "addRecipeInformation": "true",
                "fillIngredients": "true"
            }
            
            # Add query
            if analysis.get("english_query"):
                params["query"] = analysis["english_query"]
            
            # Add dietary restrictions
            if analysis.get("dietary_restrictions"):
                diet_map = {"vegetarian": "vegetarian", "vegan": "vegan", "gluten-free": "gluten free"}
                for diet in analysis["dietary_restrictions"]:
                    if diet in diet_map:
                        params["diet"] = diet_map[diet]
                        break
            
            # Add time constraint
            if analysis.get("max_cook_time"):
                params["maxReadyTime"] = analysis["max_cook_time"]
            
            async with session.get(f"{self.apis['spoonacular']['base_url']}/complexSearch", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    meals = []
                    for recipe in data.get("results", []):
                        meal = self._parse_spoonacular_recipe(recipe)
                        if meal:
                            meals.append(meal)
                    return meals
                else:
                    logger.warning(f"Spoonacular API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Spoonacular search failed: {e}")
            return []
    
    async def _search_edamam(self, session: aiohttp.ClientSession, analysis: Dict, max_results: int) -> List[Dict]:
        """Search Edamam API"""
        if not self.apis["edamam"]["enabled"]:
            return []
        
        try:
            params = {
                "type": "public",
                "app_id": self.apis["edamam"]["app_id"],
                "app_key": self.apis["edamam"]["app_key"],
                "to": max_results,
                "q": analysis.get("english_query", "dinner")
            }
            
            headers = {
                "Accept": "application/json",
                "User-Agent": "Slovenian-Grocery-Intelligence/1.0"
            }
            
            async with session.get(self.apis["edamam"]["base_url"], params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    meals = []
                    for hit in data.get("hits", []):
                        recipe = hit.get("recipe", {})
                        meal = self._parse_edamam_recipe(recipe)
                        if meal:
                            meals.append(meal)
                    return meals
                else:
                    logger.warning(f"Edamam API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Edamam search failed: {e}")
            return []
    
    async def _search_themealdb(self, session: aiohttp.ClientSession, analysis: Dict, max_results: int) -> List[Dict]:
        """Search TheMealDB API"""
        try:
            meals = []
            
            # Try searching by main ingredient if specified
            for ingredient in analysis.get("included_ingredients", [])[:2]:
                try:
                    async with session.get(f"{self.apis['themealdb']['base_url']}/filter.php?i={ingredient}") as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and data.get("meals"):
                                for meal_basic in data["meals"][:max_results//2]:
                                    detailed_meal = await self._get_themealdb_details(session, meal_basic["idMeal"])
                                    if detailed_meal:
                                        meals.append(detailed_meal)
                except Exception as e:
                    logger.warning(f"TheMealDB ingredient search failed: {e}")
            
            # If not enough results, get random meals
            if len(meals) < max_results:
                for _ in range(max_results - len(meals)):
                    try:
                        async with session.get(f"{self.apis['themealdb']['base_url']}/random.php") as response:
                            if response.status == 200:
                                data = await response.json()
                                if data and data.get("meals") and data["meals"][0]:
                                    meal = self._parse_themealdb_recipe(data["meals"][0])
                                    if meal:
                                        meals.append(meal)
                    except Exception as e:
                        logger.warning(f"TheMealDB random search failed: {e}")
            
            return meals[:max_results]
            
        except Exception as e:
            logger.error(f"TheMealDB search failed: {e}")
            return []
    
    async def _get_themealdb_details(self, session: aiohttp.ClientSession, meal_id: str) -> Optional[Dict]:
        """Get detailed meal information from TheMealDB"""
        try:
            async with session.get(f"{self.apis['themealdb']['base_url']}/lookup.php?i={meal_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    if data and data.get("meals") and data["meals"][0]:
                        return self._parse_themealdb_recipe(data["meals"][0])
            return None
        except Exception as e:
            logger.warning(f"Failed to get TheMealDB details for {meal_id}: {e}")
            return None
    
    # RECIPE PARSING METHODS
    def _parse_spoonacular_recipe(self, recipe: Dict) -> Optional[Dict]:
        """Parse Spoonacular recipe data"""
        try:
            ingredients = []
            for ing in recipe.get("extendedIngredients", []):
                ingredients.append({
                    "name": ing.get("name", ""),
                    "amount": ing.get("amount", 0),
                    "unit": ing.get("unit", ""),
                    "original": ing.get("original", "")
                })
            
            return {
                "id": f"spoon_{recipe.get('id', '')}",
                "title": recipe.get("title", "Unknown Recipe"),
                "description": recipe.get("summary", "")[:200].replace("<b>", "").replace("</b>", ""),
                "image_url": recipe.get("image", ""),
                "prep_time": recipe.get("preparationMinutes", 0) or 0,
                "cook_time": recipe.get("cookingMinutes", 0) or 0,
                "servings": recipe.get("servings", 2) or 2,
                "ingredients": ingredients,
                "instructions": recipe.get("instructions", "").split(". ") if recipe.get("instructions") else [],
                "cuisine_type": ",".join(recipe.get("cuisines", [])),
                "diet_labels": recipe.get("diets", []),
                "recipe_url": recipe.get("sourceUrl", ""),
                "source": "Spoonacular"
            }
        except Exception as e:
            logger.warning(f"Error parsing Spoonacular recipe: {e}")
            return None
    
    def _parse_edamam_recipe(self, recipe: Dict) -> Optional[Dict]:
        """Parse Edamam recipe data"""
        try:
            ingredients = []
            for ing in recipe.get("ingredients", []):
                ingredients.append({
                    "name": ing.get("food", ""),
                    "amount": ing.get("quantity", 0) or 0,
                    "unit": ing.get("measure", ""),
                    "original": ing.get("text", "")
                })
            
            return {
                "id": f"edamam_{recipe.get('uri', '').split('_')[-1] if recipe.get('uri') else 'unknown'}",
                "title": recipe.get("label", "Unknown Recipe"),
                "description": f"Delicious {recipe.get('cuisineType', ['international'])[0]} cuisine",
                "image_url": recipe.get("image", ""),
                "prep_time": 0,
                "cook_time": recipe.get("totalTime", 30) or 30,
                "servings": recipe.get("yield", 2) or 2,
                "ingredients": ingredients,
                "instructions": [],
                "cuisine_type": ",".join(recipe.get("cuisineType", [])),
                "diet_labels": recipe.get("dietLabels", []),
                "recipe_url": recipe.get("url", ""),
                "source": "Edamam"
            }
        except Exception as e:
            logger.warning(f"Error parsing Edamam recipe: {e}")
            return None
    
    def _parse_themealdb_recipe(self, meal: Dict) -> Optional[Dict]:
        """Parse TheMealDB recipe data"""
        try:
            ingredients = []
            for i in range(1, 21):
                ingredient = meal.get(f"strIngredient{i}", "")
                measure = meal.get(f"strMeasure{i}", "")
                
                if ingredient and ingredient.strip() and ingredient.strip().lower() not in ["null", "none", ""]:
                    ingredients.append({
                        "name": ingredient.strip(),
                        "amount": measure.strip() if measure else "",
                        "unit": "",
                        "original": f"{measure} {ingredient}".strip() if measure else ingredient.strip()
                    })
            
            instructions = []
            instructions_text = meal.get("strInstructions", "")
            if instructions_text:
                instructions = [inst.strip() for inst in instructions_text.split(".") if inst.strip() and len(inst.strip()) > 3]
            
            return {
                "id": f"mealdb_{meal.get('idMeal', '')}",
                "title": meal.get("strMeal", "Unknown Recipe"),
                "description": f"Traditional {meal.get('strArea', 'International')} {meal.get('strCategory', 'dish')}",
                "image_url": meal.get("strMealThumb", ""),
                "prep_time": 0,
                "cook_time": 30,
                "servings": 4,
                "ingredients": ingredients,
                "instructions": instructions,
                "cuisine_type": meal.get("strArea", "International"),
                "diet_labels": [],
                "recipe_url": meal.get("strSource", ""),
                "source": "TheMealDB"
            }
        except Exception as e:
            logger.warning(f"Error parsing TheMealDB recipe: {e}")
            return None
    
    # ADDITIONAL HELPER METHODS
    def _estimate_nutrition(self, meal: Dict) -> Dict[str, Any]:
        """Estimate basic nutrition for a meal"""
        # Simple estimation based on ingredients and meal type
        ingredient_count = len(meal.get("ingredients", []))
        servings = meal.get("servings", 2)
        
        # Base estimates
        base_calories = 200 + (ingredient_count * 50)
        calories_per_serving = base_calories / servings if servings > 0 else base_calories
        
        return {
            "calories_per_serving": round(calories_per_serving),
            "estimated_protein": "15-25g",
            "estimated_carbs": "30-50g",
            "estimated_fat": "10-20g",
            "confidence": "estimated"
        }
    
    async def _filter_and_rank_meals(self, meals: List[Dict], analysis: Dict, max_results: int) -> List[Dict]:
        """Filter and rank meals based on user preferences"""
        if not meals:
            return []
        
        # Remove duplicates by title
        seen_titles = set()
        unique_meals = []
        for meal in meals:
            title = meal.get("title", "").lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_meals.append(meal)
        
        # Score and rank meals
        scored_meals = []
        for meal in unique_meals:
            score = self._calculate_meal_score(meal, analysis)
            scored_meals.append((score, meal))
        
        # Sort by score and return top results
        scored_meals.sort(key=lambda x: x[0], reverse=True)
        return [meal for score, meal in scored_meals[:max_results]]
    
    def _calculate_meal_score(self, meal: Dict, analysis: Dict) -> float:
        """Calculate relevance score for a meal"""
        score = 0.0
        
        # Time preferences
        total_time = (meal.get("prep_time", 0) or 0) + (meal.get("cook_time", 0) or 0)
        max_time = analysis.get("max_cook_time", 60)
        
        if total_time <= max_time:
            score += 2.0
        elif total_time <= max_time * 1.5:
            score += 1.0
        
        # Ingredient matches
        meal_ingredients = [ing.get("name", "").lower() for ing in meal.get("ingredients", [])]
        for ingredient in analysis.get("included_ingredients", []):
            if any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score += 1.5
        
        # Cuisine match
        meal_cuisine = meal.get("cuisine_type", "").lower()
        for cuisine in analysis.get("cuisine_types", []):
            if cuisine.lower() in meal_cuisine:
                score += 1.0
        
        # Servings match
        target_servings = analysis.get("servings", 2)
        meal_servings = meal.get("servings", 2)
        servings_diff = abs(meal_servings - target_servings)
        if servings_diff <= 1:
            score += 1.0
        elif servings_diff <= 2:
            score += 0.5
        
        return score
    
    def _generate_meal_search_summary(self, user_request: str, meals: List[Dict], analysis: Dict) -> str:
        """Generate summary of meal search results"""
        if not meals:
            return f"No meals found for '{user_request}'. Try different search terms."
        
        total = len(meals)
        cuisines = len(set(meal.get("cuisine_type", "").split(",")[0] for meal in meals if meal.get("cuisine_type")))
        
        summary = f"Found {total} meal options for '{user_request}' across {cuisines} cuisine types. "
        
        if analysis.get("dietary_restrictions"):
            summary += f"Filtered for {', '.join(analysis['dietary_restrictions'])} dietary requirements. "
        
        avg_time = sum((meal.get("prep_time", 0) or 0) + (meal.get("cook_time", 0) or 0) for meal in meals) / total
        summary += f"Average cooking time: {avg_time:.0f} minutes."
        
        return summary
    
    # GROCERY ANALYSIS METHODS
    def _analyze_store_costs(self, ingredient_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze costs if shopping at individual stores"""
        stores = ["dm", "lidl", "mercator", "spar", "tus"]
        store_analysis = {}
        
        for store in stores:
            total_cost = 0.0
            available_items = 0
            missing_items = []
            found_products = []
            
            for ingredient, products in ingredient_results.items():
                store_product = None
                for product in products:
                    if (product.get("store_name", "").lower() == store.lower() and 
                        product.get("current_price") and 
                        float(product.get("current_price", 0)) > 0):
                        store_product = product
                        break
                
                if store_product:
                    # Convert to float to avoid Decimal addition issues
                    price = float(store_product.get("current_price", 0))
                    total_cost += price
                    available_items += 1
                    found_products.append({
                        "ingredient": ingredient,
                        "product": store_product,
                        "price": price
                    })
                else:
                    missing_items.append(ingredient)
            
            store_analysis[store] = {
                "store_name": store.upper(),
                "total_cost": round(total_cost, 2),
                "available_items": available_items,
                "missing_items": missing_items,
                "found_products": found_products,
                "completeness": round((available_items / len(ingredient_results)) * 100, 1) if ingredient_results else 0
            }
        
        return store_analysis
    
    def _analyze_combined_cheapest(self, ingredient_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze costs using cheapest option for each ingredient"""
        total_cost = 0.0
        available_items = 0
        item_details = []
        
        for ingredient, products in ingredient_results.items():
            if products:
                valid_products = [p for p in products if p.get("current_price") and float(p.get("current_price", 0)) > 0]
                if valid_products:
                    # Convert to float for comparison
                    cheapest = min(valid_products, key=lambda x: float(x.get("current_price", 0)))
                    price = float(cheapest.get("current_price", 0))
                    total_cost += price
                    available_items += 1
                    
                    item_details.append({
                        "ingredient": ingredient,
                        "price": price,
                        "store": cheapest.get("store_name", ""),
                        "product": cheapest,
                        "found": True
                    })
                else:
                    item_details.append({
                        "ingredient": ingredient,
                        "price": None,
                        "store": None,
                        "product": None,
                        "found": False
                    })
            else:
                item_details.append({
                    "ingredient": ingredient,
                    "price": None,
                    "store": None,
                    "product": None,
                    "found": False
                })
        
        return {
            "total_cost": round(total_cost, 2),
            "available_items": available_items,
            "item_details": item_details,
            "completeness": round((available_items / len(ingredient_results)) * 100, 1) if ingredient_results else 0
        }
    
    def _calculate_meal_statistics(self, meal_data: Dict, ingredient_results: Dict, combined_analysis: Dict) -> Dict[str, Any]:
        """Calculate comprehensive meal statistics"""
        servings = meal_data.get("servings", 2) or 2
        total_cost = float(combined_analysis.get("total_cost", 0))  # Ensure float conversion
        cost_per_serving = total_cost / servings if servings > 0 else total_cost
        
        return {
            "cost_per_serving": round(cost_per_serving, 2),
            "total_meal_cost": round(total_cost, 2),
            "servings": servings,
            "ingredients_found": combined_analysis.get("available_items", 0),
            "total_ingredients": len(ingredient_results),
            "grocery_completeness": combined_analysis.get("completeness", 0),
            "estimated_prep_time": (meal_data.get("prep_time", 0) or 0) + (meal_data.get("cook_time", 0) or 0),
            "difficulty": meal_data.get("difficulty", "medium")
        }
    
    def _generate_shopping_recommendations(self, store_analysis: Dict, combined_analysis: Dict) -> List[str]:
        """Generate shopping recommendations"""
        recommendations = []
        
        # Find best single store
        best_store = None
        best_completeness = 0
        for store, analysis in store_analysis.items():
            if analysis["completeness"] > best_completeness:
                best_completeness = analysis["completeness"]
                best_store = analysis["store_name"]
        
        if best_store and best_completeness >= 80:
            recommendations.append(f"Shop at {best_store} for {best_completeness}% of ingredients in one trip")
        
        # Compare costs
        combined_cost = combined_analysis.get("total_cost", 0)
        if best_store:
            single_store_cost = store_analysis[best_store.lower()]["total_cost"]
            if combined_cost < single_store_cost:
                savings = single_store_cost - combined_cost
                recommendations.append(f"Save €{savings:.2f} by shopping at multiple stores")
        
        # Add practical tips
        recommendations.append("Check for seasonal ingredients that might be on sale")
        recommendations.append("Consider buying non-perishable ingredients in bulk")
        
        return recommendations
    
    def _generate_grocery_analysis_summary(self, meal_data: Dict, store_analysis: Dict, combined_analysis: Dict) -> str:
        """Generate summary of grocery analysis"""
        meal_title = meal_data.get("title", "this meal")
        total_cost = combined_analysis.get("total_cost", 0)
        completeness = combined_analysis.get("completeness", 0)
        
        summary = f"Grocery analysis for {meal_title}: "
        summary += f"Total estimated cost €{total_cost:.2f} with {completeness:.0f}% of ingredients found in Slovenian stores. "
        
        # Find best store
        best_store = max(store_analysis.values(), key=lambda x: x["completeness"])
        summary += f"Best single store option: {best_store['store_name']} with {best_store['completeness']:.0f}% ingredient availability."
        
        return summary
    
    # REVERSE MEAL SEARCH HELPERS
    async def _suggest_meals_from_ingredients(self, ingredients: List[str]) -> List[Dict[str, Any]]:
        """Use LLM to suggest meal types based on available ingredients"""
        prompt = f"""
        Based on these available ingredients: {', '.join(ingredients)}
        
        Suggest 3-5 different types of meals that could be made with these ingredients.
        Consider different cuisines and meal types.
        
        For each suggestion, provide:
        1. Meal type (breakfast, lunch, dinner, snack)
        2. Cuisine style
        3. Specific keywords for recipe search
        4. Brief description
        
        Respond with JSON:
        [
            {{
                "meal_type": "dinner",
                "cuisine": "italian",
                "keywords": ["pasta", "chicken", "italian"],
                "description": "Italian chicken pasta dish",
                "confidence": 0.8
            }}
        ]
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
            
            suggestions = json.loads(json_text)
            return suggestions
            
        except Exception as e:
            logger.error(f"Meal suggestion generation failed: {e}")
            return [
                {
                    "meal_type": "dinner",
                    "cuisine": "international",
                    "keywords": ingredients[:3],
                    "description": "Simple meal with available ingredients",
                    "confidence": 0.5
                }
            ]
    
    async def _search_spoonacular_by_ingredients(self, session: aiohttp.ClientSession, ingredients: List[str], max_results: int) -> List[Dict]:
        """Search Spoonacular specifically by ingredients"""
        if not self.apis["spoonacular"]["enabled"]:
            return []
        
        try:
            params = {
                "apiKey": self.apis["spoonacular"]["api_key"],
                "ingredients": ",".join(ingredients[:5]),  # Limit to 5 ingredients
                "number": max_results,
                "addRecipeInformation": "true"
            }
            
            async with session.get(f"{self.apis['spoonacular']['base_url']}/findByIngredients", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    meals = []
                    for recipe in data:
                        meal = self._parse_spoonacular_ingredient_recipe(recipe)
                        if meal:
                            meals.append(meal)
                    return meals
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Spoonacular ingredient search failed: {e}")
            return []
    
    async def _search_edamam_by_ingredients(self, session: aiohttp.ClientSession, ingredients: List[str], max_results: int) -> List[Dict]:
        """Search Edamam by ingredients"""
        # Edamam doesn't have a specific ingredient search, so we'll search by ingredient names
        return await self._search_edamam(session, {"english_query": " ".join(ingredients[:3])}, max_results)
    
    def _parse_spoonacular_ingredient_recipe(self, recipe: Dict) -> Optional[Dict]:
        """Parse Spoonacular ingredient-based recipe"""
        try:
            return {
                "id": f"spoon_ing_{recipe.get('id', '')}",
                "title": recipe.get("title", "Unknown Recipe"),
                "description": f"Recipe using your available ingredients",
                "image_url": recipe.get("image", ""),
                "prep_time": 0,
                "cook_time": 30,
                "servings": 2,
                "ingredients": [{"name": ing.get("name", ""), "original": ing.get("original", "")} for ing in recipe.get("usedIngredients", [])],
                "instructions": [],
                "cuisine_type": "",
                "diet_labels": [],
                "recipe_url": "",
                "source": "Spoonacular",
                "ingredient_match_score": len(recipe.get("usedIngredients", []))
            }
        except Exception as e:
            logger.warning(f"Error parsing Spoonacular ingredient recipe: {e}")
            return None
    
    def _rank_by_ingredient_match(self, meals: List[Dict], available_ingredients: List[str]) -> List[Dict]:
        """Rank meals by how well they match available ingredients"""
        def calculate_match_score(meal):
            meal_ingredients = [ing.get("name", "").lower() for ing in meal.get("ingredients", [])]
            available_lower = [ing.lower() for ing in available_ingredients]
            
            matches = sum(1 for avail_ing in available_lower 
                         if any(avail_ing in meal_ing for meal_ing in meal_ingredients))
            
            total_meal_ingredients = len(meal_ingredients)
            match_percentage = matches / total_meal_ingredients if total_meal_ingredients > 0 else 0
            
            # Also consider existing match score from APIs
            api_score = meal.get("ingredient_match_score", 0)
            
            return match_percentage + (api_score * 0.1)
        
        # Add match scores and sort
        for meal in meals:
            meal["ingredient_match_score"] = calculate_match_score(meal)
        
        return sorted(meals, key=lambda x: x.get("ingredient_match_score", 0), reverse=True)

# Global meal searcher instance
meal_searcher = MealSearcher()

async def search_meals(user_request: str, max_results: int = 12) -> Dict[str, Any]:
    """Main function to search meals"""
    return await meal_searcher.search_meals(user_request, max_results, include_grocery_analysis=False)

async def get_meal_with_grocery_analysis(meal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to get meal with grocery analysis"""
    return await meal_searcher.get_meal_with_grocery_analysis(meal_data)

async def reverse_meal_search(available_ingredients: List[str], max_results: int = 10) -> Dict[str, Any]:
    """Main function for reverse meal search"""
    return await meal_searcher.reverse_meal_search(available_ingredients, max_results)