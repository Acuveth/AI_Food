#!/usr/bin/env python3
"""
Final Fixed Meal Search Integration - Resolves all API issues
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import os
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class MealResult:
    """Represents a meal search result"""
    id: str
    title: str
    description: str
    cuisine_type: str
    prep_time: int
    cook_time: int
    servings: int
    difficulty: str
    ingredients: List[Dict]
    nutrition: Dict
    instructions: List[str]
    image_url: Optional[str] = None
    recipe_url: Optional[str] = None
    diet_labels: List[str] = None
    allergen_info: List[str] = None
    slovenian_ingredients: List[str] = None
    grocery_shopping_list: List[Dict] = None
    estimated_cost: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MealResult to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "cuisine_type": self.cuisine_type,
            "prep_time": self.prep_time,
            "cook_time": self.cook_time,
            "servings": self.servings,
            "difficulty": self.difficulty,
            "ingredients": self.ingredients or [],
            "nutrition": self.nutrition or {},
            "instructions": self.instructions or [],
            "image_url": self.image_url,
            "recipe_url": self.recipe_url,
            "diet_labels": self.diet_labels or [],
            "allergen_info": self.allergen_info or [],
            "slovenian_ingredients": self.slovenian_ingredients or [],
            "grocery_shopping_list": self.grocery_shopping_list or [],
            "estimated_cost": self.estimated_cost
        }

class EnhancedMealSearchManager:
    """
    Final fixed meal search manager with all API issues resolved
    """
    
    def __init__(self, grocery_mcp=None):
        self.grocery_mcp = grocery_mcp
        self.session = None
        
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
                "api_key": None,  # Free API
                "enabled": True
            }
        }
        
        logger.info(f"🔧 API Status: Spoonacular={self.apis['spoonacular']['enabled']}, Edamam={self.apis['edamam']['enabled']}, TheMealDB={self.apis['themealdb']['enabled']}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search_meals_by_request(
        self, 
        user_request: str, 
        max_results: int = 16,  # Increased default from 10 to 16
        include_grocery_integration: bool = False
    ) -> Dict[str, Any]:
        """
        Main function to search meals - returns meal cards without grocery integration by default
        """
        
        logger.info(f"🍽️ Searching meals for request: '{user_request}' (max: {max_results})")
        
        # Step 1: Parse user request with AI and translate to English
        request_analysis = await self._analyze_user_meal_request(user_request)
        
        # Step 2: Search meals across APIs with increased limits
        all_meals = []
        
        # Search Spoonacular with more results
        if self.apis["spoonacular"]["enabled"]:
            try:
                spoonacular_meals = await self._search_spoonacular_meals(request_analysis, max_results//2 + 2)  # Increased allocation
                all_meals.extend(spoonacular_meals)
                logger.info(f"🥄 Spoonacular found {len(spoonacular_meals)} meals")
            except Exception as e:
                logger.error(f"Spoonacular search failed: {e}")
        
        # Search Edamam with more results
        if self.apis["edamam"]["enabled"]:
            try:
                edamam_meals = await self._search_edamam_meals(request_analysis, max_results//2 + 2)  # Increased allocation
                all_meals.extend(edamam_meals)
                logger.info(f"🍳 Edamam found {len(edamam_meals)} meals")
            except Exception as e:
                logger.error(f"Edamam search failed: {e}")
        
        # Search TheMealDB with more results
        try:
            themealdb_meals = await self._search_themealdb_meals(request_analysis, max_results//3 + 2)  # Increased allocation
            all_meals.extend(themealdb_meals)
            logger.info(f"🥘 TheMealDB found {len(themealdb_meals)} meals")
        except Exception as e:
            logger.error(f"TheMealDB search failed: {e}")
        
        # Step 3: Filter and rank meals with more aggressive filtering for better results
        filtered_meals = await self._filter_and_rank_meals(all_meals, request_analysis, max_results)
        
        # Step 4: ONLY do grocery integration if specifically requested
        if include_grocery_integration and self.grocery_mcp:
            for meal in filtered_meals:
                await self._add_grocery_integration(meal)
            logger.info(f"🛒 Added grocery integration for {len(filtered_meals)} meals")
        else:
            logger.info(f"🎴 Returning {len(filtered_meals)} meal cards without grocery integration")
        
        # Step 5: Generate presentation with enhanced info
        presentation = await self._create_meal_presentation(filtered_meals, request_analysis, user_request)
        
        # Convert MealResult objects to dictionaries for JSON serialization
        meals_as_dicts = [meal.to_dict() for meal in filtered_meals]
        
        return {
            "success": len(filtered_meals) > 0,
            "meals": meals_as_dicts,
            "presentation": presentation,
            "request_analysis": request_analysis,
            "total_found": len(all_meals),
            "filtered_count": len(filtered_meals),
            "apis_used": [api for api, config in self.apis.items() if config["enabled"]],
            "grocery_integration": include_grocery_integration,
            "search_quality": {
                "diverse_sources": len([api for api, config in self.apis.items() if config["enabled"]]),
                "result_coverage": min(100, (len(filtered_meals) / max_results) * 100),
                "api_success_rate": len([api for api, config in self.apis.items() if config["enabled"]]) / 3 * 100
            }
        }



    async def get_meal_details_with_grocery(
        self, 
        meal_id: str, 
        meal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed meal information with grocery integration for a selected meal
        """
        
        logger.info(f"🛒 Getting detailed grocery info for meal: {meal_data.get('title', 'Unknown')}")
        
        try:
            # Recreate MealResult from meal_data
            meal = MealResult(
                id=meal_data.get("id", ""),
                title=meal_data.get("title", ""),
                description=meal_data.get("description", ""),
                cuisine_type=meal_data.get("cuisine_type", ""),
                prep_time=meal_data.get("prep_time", 0),
                cook_time=meal_data.get("cook_time", 0),
                servings=meal_data.get("servings", 2),
                difficulty=meal_data.get("difficulty", "medium"),
                ingredients=meal_data.get("ingredients", []),
                nutrition=meal_data.get("nutrition", {}),
                instructions=meal_data.get("instructions", []),
                image_url=meal_data.get("image_url"),
                recipe_url=meal_data.get("recipe_url"),
                diet_labels=meal_data.get("diet_labels", []),
                allergen_info=meal_data.get("allergen_info", [])
            )
            
            # Add grocery integration
            if self.grocery_mcp:
                await self._add_grocery_integration(meal)
            
            return {
                "success": True,
                "meal": meal.to_dict(),
                "grocery_details": {
                    "slovenian_ingredients": meal.slovenian_ingredients or [],
                    "shopping_list": meal.grocery_shopping_list or [],
                    "estimated_cost": meal.estimated_cost or 0,
                    "available_stores": list(set([
                        item.get("product", {}).get("store_name", "") 
                        for item in (meal.grocery_shopping_list or [])
                        if item.get("product", {}).get("store_name")
                    ]))
                },
                "message": f"Found {len(meal.grocery_shopping_list or [])} ingredients in Slovenian stores"
            }
            
        except Exception as e:
            logger.error(f"Error getting meal details: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get grocery details for this meal"
            }



    async def _analyze_user_meal_request(self, user_request: str) -> Dict[str, Any]:
        """
        Use AI to analyze user's meal request, extract search parameters, and translate to English
        """
        
        analysis_prompt = f"""
        Analyze this meal request and extract search parameters: "{user_request}"
        
        The request might be in Slovenian. If it is, translate key terms to English for API searches.
        
        Extract information about:
        1. Meal type (breakfast, lunch, dinner, snack, dessert)
        2. Cuisine preferences (Italian, Asian, Mediterranean, etc.)
        3. Dietary restrictions (vegetarian, vegan, gluten-free, keto, etc.)
        4. Cooking time preferences (quick, medium, elaborate)
        5. Difficulty level (easy, medium, hard)
        6. Specific ingredients mentioned
        7. Number of servings needed
        8. Health preferences (healthy, comfort food, etc.)
        
        Important: Create English search keywords that will work with international recipe APIs.
        
        Examples:
        - "italijanska večerja" → search_keywords: ["italian", "dinner", "pasta"]
        - "zdrav zajtrk" → search_keywords: ["healthy", "breakfast"]
        - "hitra malica" → search_keywords: ["quick", "snack"]
        
        Respond with JSON:
        {{
            "meal_type": "dinner",
            "cuisine_types": ["italian"],
            "dietary_restrictions": [],
            "max_cook_time": 60,
            "difficulty": "medium",
            "included_ingredients": [],
            "excluded_ingredients": [],
            "servings": 2,
            "health_focus": "balanced",
            "search_keywords": ["italian", "dinner", "pasta"],
            "english_query": "Italian dinner recipes",
            "user_intent": "Italian dinner recipes",
            "original_language": "slovenian or english"
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.2,
                    max_tokens=600
                )
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            if "```json" in analysis_text:
                json_text = analysis_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = analysis_text
                
            analysis = json.loads(json_text)
            logger.info(f"🧠 Request analysis: {analysis.get('user_intent', 'Unknown intent')} (English: {analysis.get('english_query', 'Unknown')})")
            return analysis
            
        except Exception as e:
            logger.error(f"Request analysis failed: {e}")
            # Fallback analysis
            return {
                "meal_type": "dinner",
                "cuisine_types": [],
                "dietary_restrictions": [],
                "max_cook_time": 60,
                "difficulty": "medium",
                "included_ingredients": [],
                "excluded_ingredients": [],
                "servings": 2,
                "health_focus": "balanced",
                "search_keywords": [user_request],
                "english_query": user_request,
                "user_intent": user_request,
                "original_language": "unknown"
            }
    
    async def _search_spoonacular_meals(self, request_analysis: Dict, max_results: int) -> List[MealResult]:
        """Search meals using Spoonacular API with fixed parameter handling"""
        
        if not self.apis["spoonacular"]["enabled"]:
            return []
        
        try:
            # Build search parameters with proper types
            params = {
                "apiKey": self.apis["spoonacular"]["api_key"],
                "number": str(max_results),  # Convert to string
                "addRecipeInformation": "true",  # String, not boolean
                "fillIngredients": "true"  # String, not boolean
            }
            
            # Add filters based on analysis
            if request_analysis.get("dietary_restrictions"):
                diet_filters = []
                for diet in request_analysis["dietary_restrictions"]:
                    # Map common diets to Spoonacular diet types
                    diet_mapping = {
                        "vegetarian": "vegetarian",
                        "vegan": "vegan",
                        "gluten-free": "gluten free",
                        "gluten_free": "gluten free",
                        "keto": "ketogenic",
                        "paleo": "paleo"
                    }
                    if diet.lower() in diet_mapping:
                        diet_filters.append(diet_mapping[diet.lower()])
                
                if diet_filters:
                    params["diet"] = ",".join(diet_filters)
            
            if request_analysis.get("max_cook_time"):
                params["maxReadyTime"] = str(request_analysis["max_cook_time"])
            
            # Use English query for search
            english_query = request_analysis.get("english_query", "")
            if english_query:
                params["query"] = english_query
            elif request_analysis.get("search_keywords"):
                params["query"] = " ".join(request_analysis["search_keywords"])
            
            logger.info(f"🥄 Spoonacular search with query: '{params.get('query', 'No query')}'")
            
            # Make API request
            async with self.session.get(
                f"{self.apis['spoonacular']['base_url']}/complexSearch",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    meals = []
                    
                    for recipe in data.get("results", []):
                        meal = await self._parse_spoonacular_recipe(recipe)
                        meals.append(meal)
                    
                    return meals
                else:
                    logger.warning(f"Spoonacular API error: {response.status}")
                    error_text = await response.text()
                    logger.warning(f"Spoonacular error details: {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Spoonacular search failed: {e}")
            return []
    
    async def _search_edamam_meals(self, request_analysis: Dict, max_results: int) -> List[MealResult]:
        """Search meals using Edamam API with required authentication headers"""
        
        if not self.apis["edamam"]["enabled"]:
            logger.warning("🍳 Edamam API not configured - missing app_id or app_key")
            return []
        
        try:
            # Build search parameters
            params = {
                "type": "public",
                "app_id": self.apis["edamam"]["app_id"],
                "app_key": self.apis["edamam"]["app_key"],
                "to": str(max_results)
            }
            
            # Add search query - use English query
            english_query = request_analysis.get("english_query", "")
            if english_query:
                params["q"] = english_query
            elif request_analysis.get("search_keywords"):
                params["q"] = " ".join(request_analysis["search_keywords"])
            else:
                params["q"] = request_analysis.get("meal_type", "dinner")
            
            # Add diet filters
            if request_analysis.get("dietary_restrictions"):
                for diet in request_analysis["dietary_restrictions"]:
                    # Map to Edamam health labels
                    health_mapping = {
                        "vegetarian": "vegetarian",
                        "vegan": "vegan",
                        "gluten-free": "gluten-free",
                        "gluten_free": "gluten-free",
                        "dairy-free": "dairy-free",
                        "dairy_free": "dairy-free"
                    }
                    if diet.lower() in health_mapping:
                        params["health"] = health_mapping[diet.lower()]
                        break  # Edamam allows one health filter at a time
            
            # REQUIRED: Add Edamam authentication headers
            headers = {
                "Edamam-Account-User": f"user-{self.apis['edamam']['app_id']}",  # Use app_id as user identifier
                "Accept": "application/json",
                "User-Agent": "Slovenian-Grocery-Intelligence/1.0"
            }
            
            logger.info(f"🍳 Edamam search with query: '{params.get('q', 'No query')}'")
            
            # Make API request with required headers
            async with self.session.get(
                self.apis["edamam"]["base_url"],
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    meals = []
                    
                    for hit in data.get("hits", []):
                        recipe = hit.get("recipe", {})
                        meal = await self._parse_edamam_recipe(recipe)
                        if meal:  # Only add valid meals
                            meals.append(meal)
                    
                    return meals
                elif response.status == 401:
                    error_text = await response.text()
                    logger.warning(f"🍳 Edamam API authentication error: {error_text}")
                    logger.warning("🍳 Authentication failed. Possible solutions:")
                    logger.warning("🍳 1. Check your EDAMAM_APP_ID and EDAMAM_APP_KEY in .env file")
                    logger.warning("🍳 2. Ensure you selected 'Recipe Search API' when creating your Edamam app")
                    logger.warning("🍳 3. Your API credentials might need time to activate (up to 24 hours)")
                    return []
                elif response.status == 403:
                    error_text = await response.text()
                    logger.warning(f"🍳 Edamam API quota exceeded: {error_text}")
                    logger.warning("🍳 You've reached your free tier limit. Consider upgrading or trying again tomorrow.")
                    return []
                else:
                    logger.warning(f"Edamam API error: {response.status}")
                    error_text = await response.text()
                    logger.warning(f"Edamam error details: {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Edamam search failed: {e}")
            return []
    
    async def _search_themealdb_meals(self, request_analysis: Dict, max_results: int) -> List[MealResult]:
        """Search meals using TheMealDB with robust error handling"""
        
        try:
            meals = []
            
            # Map cuisine types to TheMealDB areas
            cuisine_mapping = {
                "italian": "Italian",
                "chinese": "Chinese", 
                "mexican": "Mexican",
                "indian": "Indian",
                "french": "French",
                "american": "American",
                "british": "British",
                "thai": "Thai",
                "japanese": "Japanese",
                "greek": "Greek"
            }
            
            # Search by cuisine if specified
            cuisine_types = request_analysis.get("cuisine_types", [])
            for cuisine in cuisine_types:
                if cuisine.lower() in cuisine_mapping:
                    themeal_cuisine = cuisine_mapping[cuisine.lower()]
                    try:
                        async with self.session.get(
                            f"{self.apis['themealdb']['base_url']}/filter.php?a={themeal_cuisine}",
                            timeout=10  # Add timeout
                        ) as response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    # ROBUST: Multiple null checks
                                    if (data and 
                                        isinstance(data, dict) and 
                                        "meals" in data and 
                                        data["meals"] and 
                                        isinstance(data["meals"], list)):
                                        
                                        for meal_data in data["meals"][:max_results//2]:
                                            if meal_data and "idMeal" in meal_data:
                                                full_meal = await self._get_themealdb_meal_details(meal_data["idMeal"])
                                                if full_meal:
                                                    meals.append(full_meal)
                                    else:
                                        logger.info(f"🥘 No TheMealDB meals found for cuisine: {themeal_cuisine}")
                                except Exception as json_error:
                                    logger.warning(f"TheMealDB JSON parsing error for {themeal_cuisine}: {json_error}")
                            else:
                                logger.warning(f"TheMealDB API returned status {response.status} for {themeal_cuisine}")
                    except Exception as e:
                        logger.warning(f"TheMealDB cuisine search failed for {themeal_cuisine}: {e}")
            
            # Search by main ingredient if no cuisine specified or no results
            if not meals and request_analysis.get("search_keywords"):
                for keyword in request_analysis["search_keywords"][:2]:
                    # Skip non-ingredient keywords
                    if keyword.lower() in ["dinner", "lunch", "breakfast", "quick", "healthy", "easy"]:
                        continue
                        
                    try:
                        async with self.session.get(
                            f"{self.apis['themealdb']['base_url']}/filter.php?i={keyword}",
                            timeout=10
                        ) as response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    # ROBUST: Multiple null checks
                                    if (data and 
                                        isinstance(data, dict) and 
                                        "meals" in data and 
                                        data["meals"] and 
                                        isinstance(data["meals"], list)):
                                        
                                        for meal_data in data["meals"][:max_results//3]:
                                            if meal_data and "idMeal" in meal_data:
                                                full_meal = await self._get_themealdb_meal_details(meal_data["idMeal"])
                                                if full_meal:
                                                    meals.append(full_meal)
                                    else:
                                        logger.info(f"🥘 No TheMealDB meals found for ingredient: {keyword}")
                                except Exception as json_error:
                                    logger.warning(f"TheMealDB JSON parsing error for {keyword}: {json_error}")
                    except Exception as e:
                        logger.warning(f"TheMealDB ingredient search failed for {keyword}: {e}")
            
            # Fallback: Get random meals if still no results
            if not meals:
                try:
                    for _ in range(min(3, max_results)):
                        async with self.session.get(
                            f"{self.apis['themealdb']['base_url']}/random.php",
                            timeout=10
                        ) as response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    if (data and 
                                        isinstance(data, dict) and 
                                        "meals" in data and 
                                        data["meals"] and 
                                        isinstance(data["meals"], list) and 
                                        data["meals"][0]):
                                        
                                        meal = await self._parse_themealdb_recipe(data["meals"][0])
                                        if meal:
                                            meals.append(meal)
                                except Exception as json_error:
                                    logger.warning(f"TheMealDB random meal JSON parsing error: {json_error}")
                except Exception as e:
                    logger.warning(f"TheMealDB random search failed: {e}")
            
            logger.info(f"🥘 TheMealDB successfully found {len(meals)} meals")
            return meals[:max_results]
            
        except Exception as e:
            logger.error(f"TheMealDB search failed with error: {e}")
            return []
    
    async def _get_themealdb_meal_details(self, meal_id: str) -> Optional[MealResult]:
        """Get full meal details from TheMealDB with robust error handling"""
        try:
            if not meal_id:
                return None
                
            async with self.session.get(
                f"{self.apis['themealdb']['base_url']}/lookup.php?i={meal_id}",
                timeout=10
            ) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        # ROBUST: Multiple null checks
                        if (data and 
                            isinstance(data, dict) and 
                            "meals" in data and 
                            data["meals"] and 
                            isinstance(data["meals"], list) and 
                            len(data["meals"]) > 0 and 
                            data["meals"][0] and
                            isinstance(data["meals"][0], dict)):
                            
                            meal_data = data["meals"][0]
                            return await self._parse_themealdb_recipe(meal_data)
                        else:
                            logger.warning(f"Invalid meal data structure for meal_id {meal_id}")
                            return None
                    except Exception as json_error:
                        logger.warning(f"JSON parsing error for meal_id {meal_id}: {json_error}")
                        return None
                else:
                    logger.warning(f"TheMealDB API returned status {response.status} for meal_id {meal_id}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to get TheMealDB meal details for {meal_id}: {e}")
            return None
    
    async def _parse_spoonacular_recipe(self, recipe: Dict) -> MealResult:
        """Parse Spoonacular recipe data with null safety"""
        
        if not recipe or not isinstance(recipe, dict):
            return None
        
        ingredients = []
        for ing in recipe.get("extendedIngredients", []):
            if isinstance(ing, dict):
                ingredients.append({
                    "name": ing.get("name", ""),
                    "amount": ing.get("amount", 0),
                    "unit": ing.get("unit", ""),
                    "original": ing.get("original", "")
                })
        
        # Safe extraction of summary text
        summary = recipe.get("summary", "")
        if summary:
            # Remove HTML tags
            summary = summary.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
            summary = summary[:200]
        
        return MealResult(
            id=f"spoon_{recipe.get('id', '')}",
            title=recipe.get("title", "Unknown Recipe"),
            description=summary,
            cuisine_type=",".join(recipe.get("cuisines", [])),
            prep_time=recipe.get("preparationMinutes", 0) or 0,
            cook_time=recipe.get("cookingMinutes", 0) or 0,
            servings=recipe.get("servings", 2) or 2,
            difficulty="medium",
            ingredients=ingredients,
            nutrition={
                "calories": recipe.get("nutrition", {}).get("calories", 0) or 0,
                "protein": recipe.get("nutrition", {}).get("protein", "0g") or "0g",
                "fat": recipe.get("nutrition", {}).get("fat", "0g") or "0g",
                "carbs": recipe.get("nutrition", {}).get("carbohydrates", "0g") or "0g"
            },
            instructions=recipe.get("instructions", "").split(". ") if recipe.get("instructions") else [],
            image_url=recipe.get("image", ""),
            recipe_url=recipe.get("sourceUrl", ""),
            diet_labels=recipe.get("diets", []) or [],
            allergen_info=[]
        )
    
    async def _parse_edamam_recipe(self, recipe: Dict) -> MealResult:
        """Parse Edamam recipe data with null safety"""
        
        if not recipe or not isinstance(recipe, dict):
            return None
        
        ingredients = []
        for ing in recipe.get("ingredients", []):
            if isinstance(ing, dict):
                ingredients.append({
                    "name": ing.get("food", ""),
                    "amount": ing.get("quantity", 0) or 0,
                    "unit": ing.get("measure", ""),
                    "original": ing.get("text", "")
                })
        
        # Safe nutrition extraction
        total_nutrients = recipe.get("totalNutrients", {})
        protein_data = total_nutrients.get("PROCNT", {})
        fat_data = total_nutrients.get("FAT", {})
        carb_data = total_nutrients.get("CHOCDF", {})
        
        return MealResult(
            id=f"edamam_{recipe.get('uri', '').split('_')[-1] if recipe.get('uri') else 'unknown'}",
            title=recipe.get("label", "Unknown Recipe"),
            description=f"Delicious {recipe.get('cuisineType', ['international'])[0]} cuisine",
            cuisine_type=",".join(recipe.get("cuisineType", [])),
            prep_time=0,
            cook_time=recipe.get("totalTime", 0) or 30,
            servings=recipe.get("yield", 2) or 2,
            difficulty="medium",
            ingredients=ingredients,
            nutrition={
                "calories": recipe.get("calories", 0) or 0,
                "protein": f"{protein_data.get('quantity', 0) or 0:.1f}g",
                "fat": f"{fat_data.get('quantity', 0) or 0:.1f}g",
                "carbs": f"{carb_data.get('quantity', 0) or 0:.1f}g"
            },
            instructions=[],
            image_url=recipe.get("image", ""),
            recipe_url=recipe.get("url", ""),
            diet_labels=recipe.get("dietLabels", []) or [],
            allergen_info=recipe.get("cautions", []) or []
        )
    

    async def _calculate_enhanced_meal_score(self, meal: MealResult, request_analysis: Dict) -> float:
        """Enhanced scoring algorithm for better meal ranking"""
        
        score = 0.0
        
        # Base score for having key information
        if meal.title:
            score += 1.0
        if meal.ingredients and len(meal.ingredients) > 0:
            score += 1.0
        if meal.image_url:
            score += 0.5
        
        # Time preferences - with null safety and better scoring
        prep_time = meal.prep_time or 0
        cook_time = meal.cook_time or 0
        total_time = prep_time + cook_time
        max_time = request_analysis.get("max_cook_time") or 60
        
        if total_time <= max_time:
            score += 3.0  # Higher weight for time match
        elif total_time <= max_time * 1.2:
            score += 2.0
        elif total_time <= max_time * 1.5:
            score += 1.0
        
        # Dietary restrictions match - higher weight
        meal_diets = [d.lower() for d in (meal.diet_labels or [])]
        user_diets = [d.lower() for d in request_analysis.get("dietary_restrictions", [])]
        for diet in user_diets:
            if any(diet in meal_diet for meal_diet in meal_diets):
                score += 3.0  # High weight for diet compatibility
        
        # Cuisine preference - enhanced matching
        user_cuisines = [c.lower() for c in request_analysis.get("cuisine_types", [])]
        meal_cuisine = (meal.cuisine_type or "").lower()
        for cuisine in user_cuisines:
            if cuisine in meal_cuisine or meal_cuisine in cuisine:
                score += 2.5  # Good weight for cuisine match
        
        # Ingredient preferences - enhanced logic
        meal_ingredients = []
        if meal.ingredients:
            for ing in meal.ingredients:
                if isinstance(ing, dict) and "name" in ing and ing["name"]:
                    meal_ingredients.append(ing["name"].lower())
        
        # Bonus for included ingredients
        for ingredient in request_analysis.get("included_ingredients", []):
            if ingredient and any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score += 1.5
        
        # Penalty for excluded ingredients
        for ingredient in request_analysis.get("excluded_ingredients", []):
            if ingredient and any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score -= 3.0  # Strong penalty for excluded ingredients
        
        # Servings match - enhanced scoring
        target_servings = request_analysis.get("servings") or 2
        meal_servings = meal.servings or 2
        servings_diff = abs(meal_servings - target_servings)
        if servings_diff == 0:
            score += 1.5
        elif servings_diff <= 2:
            score += 1.0
        elif servings_diff <= 4:
            score += 0.5
        
        # Bonus for complete recipes (have instructions)
        if meal.instructions and len(meal.instructions) > 0:
            score += 1.0
        
        # Bonus for nutritional information
        if meal.nutrition and len(meal.nutrition) > 0:
            score += 0.5
        
        # Health focus bonus
        health_focus = request_analysis.get("health_focus", "").lower()
        if health_focus == "healthy":
            # Look for healthy indicators
            healthy_keywords = ["vegetable", "fruit", "lean", "grilled", "baked", "steamed"]
            meal_text = f"{meal.title} {' '.join([ing.get('name', '') for ing in (meal.ingredients or [])])}"
            if any(keyword in meal_text.lower() for keyword in healthy_keywords):
                score += 1.5
        
        return max(0.0, score)  # Ensure non-negative score


    def _ensure_cuisine_diversity(self, scored_meals: List[Tuple[float, MealResult]], max_results: int) -> List[MealResult]:
        """Ensure diversity in cuisine types for better user experience"""
        
        selected_meals = []
        cuisine_counts = {}
        max_per_cuisine = max(2, max_results // 4)  # Max 2-4 meals per cuisine type
        
        for score, meal in scored_meals:
            if len(selected_meals) >= max_results:
                break
                
            cuisine = (meal.cuisine_type or 'international').lower()
            current_count = cuisine_counts.get(cuisine, 0)
            
            # Add meal if we haven't reached the limit for this cuisine
            if current_count < max_per_cuisine:
                selected_meals.append(meal)
                cuisine_counts[cuisine] = current_count + 1
        
        # If we still need more meals and have room, add remaining high-scoring meals
        if len(selected_meals) < max_results:
            for score, meal in scored_meals:
                if len(selected_meals) >= max_results:
                    break
                if meal not in selected_meals:
                    selected_meals.append(meal)
        
        return selected_meals

    async def _parse_themealdb_recipe(self, meal: Dict) -> MealResult:
        """Parse TheMealDB recipe data with comprehensive null checks"""
        
        # ROBUST: Handle None or invalid meal data
        if not meal or not isinstance(meal, dict):
            return None
            
        # Extract basic information with defaults
        meal_id = meal.get("idMeal", "")
        meal_title = meal.get("strMeal", "Unknown Recipe")
        meal_area = meal.get("strArea", "International")
        meal_category = meal.get("strCategory", "dish")
        
        ingredients = []
        for i in range(1, 21):  # TheMealDB has ingredients 1-20
            ingredient = meal.get(f"strIngredient{i}", "")
            measure = meal.get(f"strMeasure{i}", "")
            
            # Skip empty, null, or "null" string ingredients
            if (ingredient and 
                ingredient.strip() and 
                ingredient.strip().lower() not in ["null", "none", ""]):
                
                measure_clean = measure.strip() if measure and measure.strip() else ""
                ingredients.append({
                    "name": ingredient.strip(),
                    "amount": measure_clean,
                    "unit": "",
                    "original": f"{measure_clean} {ingredient}".strip() if measure_clean else ingredient.strip()
                })
        
        instructions = []
        instructions_text = meal.get("strInstructions", "")
        if instructions_text and instructions_text.strip():
            instructions = [
                inst.strip() 
                for inst in instructions_text.split(".") 
                if inst.strip() and len(inst.strip()) > 3
            ]
        
        # Create MealResult with null-safe defaults
        return MealResult(
            id=f"mealdb_{meal_id}",
            title=meal_title,
            description=f"Traditional {meal_area} {meal_category}",
            cuisine_type=meal_area,
            prep_time=0,  # Default values instead of None
            cook_time=30,  # Default estimate
            servings=4,
            difficulty="medium",
            ingredients=ingredients,
            nutrition={},
            instructions=instructions,
            image_url=meal.get("strMealThumb", ""),
            recipe_url=meal.get("strSource", ""),
            diet_labels=[],
            allergen_info=[]
        )
    
    async def _filter_and_rank_meals(
        self, 
        meals: List[MealResult], 
        request_analysis: Dict, 
        max_results: int
    ) -> List[MealResult]:
        """Filter and rank meals based on user preferences with improved algorithm"""
        
        # FIRST: Filter out None meals
        valid_meals = [meal for meal in meals if meal is not None]
        
        if not valid_meals:
            return []
        
        # Remove duplicates based on title similarity with more aggressive deduplication
        unique_meals = []
        seen_titles = set()
        
        for meal in valid_meals:
            if not meal.title:  # Skip meals without titles
                continue
                
            title_lower = meal.title.lower()
            # More aggressive duplicate detection
            is_duplicate = any(
                self._calculate_similarity(title_lower, seen_title) > 0.7  # Lowered threshold for better variety
                for seen_title in seen_titles
            )
            
            if not is_duplicate:
                unique_meals.append(meal)
                seen_titles.add(title_lower)
        
        # Score meals based on user preferences with enhanced scoring
        scored_meals = []
        for meal in unique_meals:
            try:
                score = await self._calculate_enhanced_meal_score(meal, request_analysis)
                scored_meals.append((score, meal))
            except Exception as e:
                logger.warning(f"Error scoring meal '{meal.title}': {e}")
                # Include with default score if scoring fails
                scored_meals.append((0.0, meal))
        
        # Sort by score and return top results
        scored_meals.sort(key=lambda x: x[0], reverse=True)
        
        # Ensure diverse results by limiting similar cuisine types
        diverse_meals = self._ensure_cuisine_diversity(scored_meals, max_results)
        
        return diverse_meals
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _calculate_meal_score(self, meal: MealResult, request_analysis: Dict) -> float:
        """Calculate relevance score for a meal with null safety"""
        
        score = 0.0
        
        # Time preferences - with null safety
        prep_time = meal.prep_time or 0
        cook_time = meal.cook_time or 0
        total_time = prep_time + cook_time
        max_time = request_analysis.get("max_cook_time") or 60
        
        if total_time <= max_time:
            score += 2.0
        elif total_time <= max_time * 1.5:
            score += 1.0
        
        # Dietary restrictions match - with null safety
        meal_diets = [d.lower() for d in (meal.diet_labels or [])]
        user_diets = [d.lower() for d in request_analysis.get("dietary_restrictions", [])]
        for diet in user_diets:
            if any(diet in meal_diet for meal_diet in meal_diets):
                score += 2.0
        
        # Cuisine preference - with null safety
        user_cuisines = [c.lower() for c in request_analysis.get("cuisine_types", [])]
        meal_cuisine = (meal.cuisine_type or "").lower()
        for cuisine in user_cuisines:
            if cuisine in meal_cuisine:
                score += 1.5
        
        # Ingredient preferences - with null safety
        meal_ingredients = []
        if meal.ingredients:
            for ing in meal.ingredients:
                if isinstance(ing, dict) and "name" in ing and ing["name"]:
                    meal_ingredients.append(ing["name"].lower())
        
        # Bonus for included ingredients
        for ingredient in request_analysis.get("included_ingredients", []):
            if ingredient and any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score += 1.0
        
        # Penalty for excluded ingredients
        for ingredient in request_analysis.get("excluded_ingredients", []):
            if ingredient and any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score -= 2.0
        
        # Servings match - with null safety
        target_servings = request_analysis.get("servings") or 2
        meal_servings = meal.servings or 2
        servings_diff = abs(meal_servings - target_servings)
        if servings_diff == 0:
            score += 1.0
        elif servings_diff <= 2:
            score += 0.5
        
        return max(0.0, score)  # Ensure non-negative score
    
    async def _add_grocery_integration(self, meal: MealResult) -> None:
        """Add grocery shopping integration for a meal"""
        
        if not self.grocery_mcp:
            return
        
        try:
            # Translate ingredients to Slovenian and find in grocery database
            slovenian_ingredients = []
            shopping_list = []
            total_cost = 0.0
            
            for ingredient in meal.ingredients[:10]:  # Limit to 10 ingredients
                # Translate ingredient name to Slovenian
                slovenian_name = await self._translate_ingredient_to_slovenian(ingredient["name"])
                slovenian_ingredients.append(slovenian_name)
                
                # Search in grocery database
                try:
                    products = await self.grocery_mcp.find_cheapest_product(
                        slovenian_name, use_semantic_validation=True
                    )
                    
                    if products:
                        best_product = products[0]  # Get cheapest
                        shopping_list.append({
                            "ingredient": ingredient["name"],
                            "slovenian_name": slovenian_name,
                            "product": best_product,
                            "needed_amount": ingredient["original"],
                            "estimated_cost": best_product.get("current_price", 0)
                        })
                        total_cost += best_product.get("current_price", 0)
                
                except Exception as e:
                    logger.warning(f"Failed to find grocery product for {slovenian_name}: {e}")
            
            # Update meal with grocery information
            meal.slovenian_ingredients = slovenian_ingredients
            meal.grocery_shopping_list = shopping_list
            meal.estimated_cost = total_cost
            
            logger.info(f"🛒 Added grocery integration for '{meal.title}' - estimated cost: €{total_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Grocery integration failed for meal '{meal.title}': {e}")
    
    async def _translate_ingredient_to_slovenian(self, ingredient: str) -> str:
        """Translate ingredient name to Slovenian"""
        
        # Common translations cache
        translations = {
            "chicken": "piščanec",
            "beef": "goveje meso",
            "pork": "svinjina",
            "fish": "riba",
            "tomato": "paradižnik",
            "onion": "čebula",
            "garlic": "česen",
            "potato": "krompir",
            "rice": "riž",
            "pasta": "testenine",
            "cheese": "sir",
            "milk": "mleko",
            "egg": "jajce",
            "bread": "kruh",
            "oil": "olje",
            "salt": "sol",
            "pepper": "poper",
            "sugar": "sladkor",
            "flour": "moka",
            "butter": "maslo",
            "olive oil": "oljčno olje",
            "parmesan": "parmezan",
            "mozzarella": "mocarela",
            "basil": "bazilika",
            "oregano": "oregano"
        }
        
        ingredient_lower = ingredient.lower().strip()
        
        # Check cache first
        if ingredient_lower in translations:
            return translations[ingredient_lower]
        
        # Use AI translation for unknown ingredients
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user", 
                        "content": f"Translate this cooking ingredient to Slovenian (just the translation, no explanation): {ingredient}"
                    }],
                    temperature=0.1,
                    max_tokens=50
                )
            )
            
            translation = response.choices[0].message.content.strip()
            return translation
            
        except Exception as e:
            logger.warning(f"Translation failed for '{ingredient}': {e}")
            return ingredient  # Return original if translation fails
    
    async def _create_meal_presentation(
        self, 
        meals: List[MealResult], 
        request_analysis: Dict, 
        original_request: str
    ) -> Dict[str, Any]:
        """Create a comprehensive presentation of meal results"""
        
        if not meals:
            return {
                "summary": f"No meals found for '{original_request}'. Try different search terms or check if APIs are configured.",
                "suggestions": [
                    "Try broader search terms like 'dinner' or 'pasta'",
                    "Check if Spoonacular and Edamam API keys are configured",
                    "Use English terms like 'Italian dinner' instead of 'italijanska večerja'"
                ],
                "meal_cards": []
            }
        
        # Create meal cards
        meal_cards = []
        for i, meal in enumerate(meals, 1):
            card = {
                "rank": i,
                "title": meal.title,
                "description": meal.description[:200] + "..." if len(meal.description) > 200 else meal.description,
                "cuisine": meal.cuisine_type,
                "time_info": {
                    "prep_time": meal.prep_time,
                    "cook_time": meal.cook_time,
                    "total_time": meal.prep_time + meal.cook_time
                },
                "servings": meal.servings,
                "difficulty": meal.difficulty,
                "dietary_info": {
                    "diet_labels": meal.diet_labels,
                    "allergens": meal.allergen_info
                },
                "nutrition": meal.nutrition,
                "image_url": meal.image_url,
                "recipe_url": meal.recipe_url,
                "grocery_integration": {
                    "available": bool(meal.grocery_shopping_list),
                    "estimated_cost": meal.estimated_cost,
                    "ingredient_count": len(meal.grocery_shopping_list) if meal.grocery_shopping_list else 0
                },
                "key_ingredients": [ing["name"] for ing in meal.ingredients[:5]]
            }
            meal_cards.append(card)
        
        # Create summary
        summary = await self._generate_meal_summary(meals, request_analysis, original_request)
        
        return {
            "summary": summary,
            "total_meals": len(meals),
            "user_request": original_request,
            "search_criteria": request_analysis,
            "meal_cards": meal_cards,
            "recommendations": self._generate_recommendations(meals, request_analysis)
        }
    
    async def _generate_meal_summary(
        self, 
        meals: List[MealResult], 
        request_analysis: Dict, 
        original_request: str
    ) -> str:
        """Generate AI summary of meal search results"""
        
        meal_info = []
        for meal in meals[:5]:  # Summarize top 5
            meal_info.append({
                "title": meal.title,
                "cuisine": meal.cuisine_type,
                "time": meal.prep_time + meal.cook_time,
                "diet_labels": meal.diet_labels
            })
        
        summary_prompt = f"""
        Create a helpful summary for a user who searched for: "{original_request}"
        
        User wanted: {request_analysis.get('user_intent', 'meal recommendations')}
        
        Found meals:
        {json.dumps(meal_info, indent=2)}
        
        Create a 2-3 sentence summary that:
        1. Confirms what was found
        2. Highlights variety or special features
        3. Mentions any standout options
        
        Keep it friendly and helpful. Respond in the same language as the original request.
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=150
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Found {len(meals)} meal options for your request '{original_request}'. The selection includes various cuisines and cooking times to match your preferences."
    
    def _generate_recommendations(
        self, 
        meals: List[MealResult], 
        request_analysis: Dict
    ) -> List[str]:
        """Generate helpful recommendations based on search results with null safety"""
        
        recommendations = []
        
        if not meals:
            recommendations.extend([
                "Try using English search terms (e.g., 'Italian dinner' instead of 'italijanska večerja')",
                "Check if your API keys are configured correctly",
                "Use broader search terms like 'pasta' or 'chicken'"
            ])
            return recommendations
        
        # Time-based recommendations - with null safety
        quick_meals = []
        for m in meals:
            prep_time = m.prep_time or 0
            cook_time = m.cook_time or 0
            total_time = prep_time + cook_time
            if total_time <= 30:
                quick_meals.append(m)
        
        if quick_meals and (request_analysis.get("max_cook_time") or 60) > 30:
            recommendations.append(f"For quick options, try: {quick_meals[0].title}")
        
        # Diet-specific recommendations - with null safety
        diet_meals = []
        for m in meals:
            meal_diets = m.diet_labels or []
            user_diets = request_analysis.get("dietary_restrictions", [])
            if any(diet in meal_diets for diet in user_diets):
                diet_meals.append(m)
        
        if diet_meals:
            recommendations.append(f"Perfect for your diet: {diet_meals[0].title}")
        
        # Budget recommendations - with null safety
        meals_with_cost = [m for m in meals if m.estimated_cost and m.estimated_cost > 0]
        if meals_with_cost:
            cheapest = min(meals_with_cost, key=lambda x: x.estimated_cost or 0)
            recommendations.append(f"Most budget-friendly: {cheapest.title} (€{cheapest.estimated_cost:.2f})")
        
        # Grocery integration recommendations - with null safety
        grocery_meals = [m for m in meals if m.grocery_shopping_list]
        if grocery_meals:
            recommendations.append(f"Complete shopping list available for: {grocery_meals[0].title}")
        
        return recommendations[:4]  # Limit to 4 recommendations

# Usage example and integration
async def search_meals_for_user(
    user_request: str, 
    grocery_mcp=None, 
    max_results: int = 8
) -> Dict[str, Any]:
    """
    Main function to search meals and integrate with grocery system
    """
    async with EnhancedMealSearchManager(grocery_mcp) as meal_manager:
        result = await meal_manager.search_meals_by_request(
            user_request=user_request,
            max_results=max_results,
            include_grocery_integration=bool(grocery_mcp)
        )
        
        return result