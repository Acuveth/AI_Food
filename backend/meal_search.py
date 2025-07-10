#!/usr/bin/env python3
"""
Enhanced Meal Search Integration for Slovenian Grocery Intelligence
Integrates meal APIs with grocery shopping recommendations
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
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

class EnhancedMealSearchManager:
    """
    Manages meal search across multiple APIs and integrates with grocery system
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
        max_results: int = 10,
        include_grocery_integration: bool = True
    ) -> Dict[str, Any]:
        """
        Main function to search meals based on user request and integrate with grocery system
        """
        
        logger.info(f"🍽️ Searching meals for request: '{user_request}'")
        
        # Step 1: Parse user request with AI
        request_analysis = await self._analyze_user_meal_request(user_request)
        
        # Step 2: Search meals across APIs
        all_meals = []
        
        # Search Spoonacular
        if self.apis["spoonacular"]["enabled"]:
            spoonacular_meals = await self._search_spoonacular_meals(request_analysis, max_results//2)
            all_meals.extend(spoonacular_meals)
        
        # Search Edamam
        if self.apis["edamam"]["enabled"]:
            edamam_meals = await self._search_edamam_meals(request_analysis, max_results//2)
            all_meals.extend(edamam_meals)
        
        # Search TheMealDB (free backup)
        themealdb_meals = await self._search_themealdb_meals(request_analysis, max_results//3)
        all_meals.extend(themealdb_meals)
        
        # Step 3: Filter and rank meals
        filtered_meals = await self._filter_and_rank_meals(all_meals, request_analysis, max_results)
        
        # Step 4: Integrate with grocery system
        if include_grocery_integration and self.grocery_mcp:
            for meal in filtered_meals:
                await self._add_grocery_integration(meal)
        
        # Step 5: Generate presentation
        presentation = await self._create_meal_presentation(filtered_meals, request_analysis, user_request)
        
        return {
            "success": True,
            "meals": filtered_meals,
            "presentation": presentation,
            "request_analysis": request_analysis,
            "total_found": len(all_meals),
            "filtered_count": len(filtered_meals),
            "apis_used": [api for api, config in self.apis.items() if config["enabled"]],
            "grocery_integration": include_grocery_integration
        }
    
    async def _analyze_user_meal_request(self, user_request: str) -> Dict[str, Any]:
        """
        Use AI to analyze user's meal request and extract search parameters
        """
        
        analysis_prompt = f"""
        Analyze this meal request and extract search parameters: "{user_request}"
        
        Extract information about:
        1. Meal type (breakfast, lunch, dinner, snack, dessert)
        2. Cuisine preferences (Italian, Asian, Mediterranean, etc.)
        3. Dietary restrictions (vegetarian, vegan, gluten-free, keto, etc.)
        4. Cooking time preferences (quick, medium, elaborate)
        5. Difficulty level (easy, medium, hard)
        6. Specific ingredients mentioned
        7. Number of servings needed
        8. Health preferences (healthy, comfort food, etc.)
        9. Occasion (family dinner, date night, meal prep, etc.)
        10. Budget considerations
        
        Respond with JSON:
        {{
            "meal_type": "dinner",
            "cuisine_types": ["italian", "mediterranean"],
            "dietary_restrictions": ["vegetarian"],
            "max_cook_time": 30,
            "difficulty": "easy",
            "included_ingredients": ["tomatoes", "pasta"],
            "excluded_ingredients": ["meat"],
            "servings": 4,
            "health_focus": "healthy",
            "occasion": "family_dinner",
            "budget": "moderate",
            "search_keywords": ["pasta", "vegetarian", "italian"],
            "user_intent": "Quick vegetarian Italian dinner for family"
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
            logger.info(f"🧠 Request analysis: {analysis.get('user_intent', 'Unknown intent')}")
            return analysis
            
        except Exception as e:
            logger.error(f"Request analysis failed: {e}")
            return {
                "meal_type": "any",
                "cuisine_types": [],
                "dietary_restrictions": [],
                "max_cook_time": 60,
                "difficulty": "any",
                "included_ingredients": [],
                "excluded_ingredients": [],
                "servings": 2,
                "health_focus": "balanced",
                "occasion": "general",
                "budget": "moderate",
                "search_keywords": [user_request],
                "user_intent": user_request
            }
    
    async def _search_spoonacular_meals(self, request_analysis: Dict, max_results: int) -> List[MealResult]:
        """Search meals using Spoonacular API"""
        
        if not self.apis["spoonacular"]["enabled"]:
            return []
        
        try:
            # Build search parameters
            params = {
                "apiKey": self.apis["spoonacular"]["api_key"],
                "number": max_results,
                "addRecipeInformation": True,
                "fillIngredients": True
            }
            
            # Add filters based on analysis
            if request_analysis.get("dietary_restrictions"):
                params["diet"] = ",".join(request_analysis["dietary_restrictions"])
            
            if request_analysis.get("max_cook_time"):
                params["maxReadyTime"] = request_analysis["max_cook_time"]
            
            if request_analysis.get("search_keywords"):
                params["query"] = " ".join(request_analysis["search_keywords"])
            
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
                    
                    logger.info(f"🥄 Spoonacular found {len(meals)} meals")
                    return meals
                else:
                    logger.warning(f"Spoonacular API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Spoonacular search failed: {e}")
            return []
    
    async def _search_edamam_meals(self, request_analysis: Dict, max_results: int) -> List[MealResult]:
        """Search meals using Edamam API"""
        
        if not self.apis["edamam"]["enabled"]:
            return []
        
        try:
            # Build search parameters
            params = {
                "type": "public",
                "app_id": self.apis["edamam"]["app_id"],
                "app_key": self.apis["edamam"]["app_key"],
                "to": max_results
            }
            
            # Add search query
            if request_analysis.get("search_keywords"):
                params["q"] = " ".join(request_analysis["search_keywords"])
            else:
                params["q"] = request_analysis.get("meal_type", "dinner")
            
            # Add diet filters
            if request_analysis.get("dietary_restrictions"):
                for diet in request_analysis["dietary_restrictions"]:
                    if diet in ["vegetarian", "vegan", "gluten-free", "dairy-free"]:
                        params[f"health"] = diet
            
            # Make API request
            async with self.session.get(
                self.apis["edamam"]["base_url"],
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    meals = []
                    
                    for hit in data.get("hits", []):
                        recipe = hit.get("recipe", {})
                        meal = await self._parse_edamam_recipe(recipe)
                        meals.append(meal)
                    
                    logger.info(f"🍳 Edamam found {len(meals)} meals")
                    return meals
                else:
                    logger.warning(f"Edamam API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Edamam search failed: {e}")
            return []
    
    async def _search_themealdb_meals(self, request_analysis: Dict, max_results: int) -> List[MealResult]:
        """Search meals using TheMealDB (free API)"""
        
        try:
            meals = []
            
            # Search by main ingredient if mentioned
            if request_analysis.get("included_ingredients"):
                for ingredient in request_analysis["included_ingredients"][:2]:  # Limit to 2 ingredients
                    async with self.session.get(
                        f"{self.apis['themealdb']['base_url']}/filter.php?i={ingredient}"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            for meal in data.get("meals", [])[:max_results//2]:
                                parsed_meal = await self._parse_themealdb_recipe(meal)
                                meals.append(parsed_meal)
            
            # Search by category/cuisine
            cuisines = ["Italian", "Chinese", "Mexican", "Indian", "French"]
            for cuisine in cuisines[:2]:
                async with self.session.get(
                    f"{self.apis['themealdb']['base_url']}/filter.php?a={cuisine}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for meal in data.get("meals", [])[:max_results//4]:
                            parsed_meal = await self._parse_themealdb_recipe(meal)
                            meals.append(parsed_meal)
            
            logger.info(f"🥘 TheMealDB found {len(meals)} meals")
            return meals[:max_results]
            
        except Exception as e:
            logger.error(f"TheMealDB search failed: {e}")
            return []
    
    async def _parse_spoonacular_recipe(self, recipe: Dict) -> MealResult:
        """Parse Spoonacular recipe data"""
        
        ingredients = []
        for ing in recipe.get("extendedIngredients", []):
            ingredients.append({
                "name": ing.get("name", ""),
                "amount": ing.get("amount", 0),
                "unit": ing.get("unit", ""),
                "original": ing.get("original", "")
            })
        
        return MealResult(
            id=f"spoon_{recipe.get('id', '')}",
            title=recipe.get("title", ""),
            description=recipe.get("summary", "").replace("<b>", "").replace("</b>", ""),
            cuisine_type=",".join(recipe.get("cuisines", [])),
            prep_time=recipe.get("preparationMinutes", 0),
            cook_time=recipe.get("cookingMinutes", 0),
            servings=recipe.get("servings", 2),
            difficulty="medium",
            ingredients=ingredients,
            nutrition={
                "calories": recipe.get("nutrition", {}).get("calories", 0),
                "protein": recipe.get("nutrition", {}).get("protein", ""),
                "fat": recipe.get("nutrition", {}).get("fat", ""),
                "carbs": recipe.get("nutrition", {}).get("carbohydrates", "")
            },
            instructions=recipe.get("instructions", "").split(". ") if recipe.get("instructions") else [],
            image_url=recipe.get("image", ""),
            recipe_url=recipe.get("sourceUrl", ""),
            diet_labels=recipe.get("diets", []),
            allergen_info=[]
        )
    
    async def _parse_edamam_recipe(self, recipe: Dict) -> MealResult:
        """Parse Edamam recipe data"""
        
        ingredients = []
        for ing in recipe.get("ingredients", []):
            ingredients.append({
                "name": ing.get("food", ""),
                "amount": ing.get("quantity", 0),
                "unit": ing.get("measure", ""),
                "original": ing.get("text", "")
            })
        
        return MealResult(
            id=f"edamam_{recipe.get('uri', '').split('_')[-1] if recipe.get('uri') else ''}",
            title=recipe.get("label", ""),
            description=f"Delicious {recipe.get('cuisineType', [''])[0]} cuisine",
            cuisine_type=",".join(recipe.get("cuisineType", [])),
            prep_time=0,
            cook_time=recipe.get("totalTime", 0),
            servings=recipe.get("yield", 2),
            difficulty="medium",
            ingredients=ingredients,
            nutrition={
                "calories": recipe.get("calories", 0),
                "protein": recipe.get("totalNutrients", {}).get("PROCNT", {}).get("quantity", 0),
                "fat": recipe.get("totalNutrients", {}).get("FAT", {}).get("quantity", 0),
                "carbs": recipe.get("totalNutrients", {}).get("CHOCDF", {}).get("quantity", 0)
            },
            instructions=[],
            image_url=recipe.get("image", ""),
            recipe_url=recipe.get("url", ""),
            diet_labels=recipe.get("dietLabels", []),
            allergen_info=recipe.get("cautions", [])
        )
    
    async def _parse_themealdb_recipe(self, meal: Dict) -> MealResult:
        """Parse TheMealDB recipe data"""
        
        ingredients = []
        for i in range(1, 21):  # TheMealDB has ingredients 1-20
            ingredient = meal.get(f"strIngredient{i}", "")
            measure = meal.get(f"strMeasure{i}", "")
            if ingredient and ingredient.strip():
                ingredients.append({
                    "name": ingredient,
                    "amount": measure,
                    "unit": "",
                    "original": f"{measure} {ingredient}".strip()
                })
        
        instructions = meal.get("strInstructions", "").split(". ") if meal.get("strInstructions") else []
        
        return MealResult(
            id=f"mealdb_{meal.get('idMeal', '')}",
            title=meal.get("strMeal", ""),
            description=f"Traditional {meal.get('strArea', '')} {meal.get('strCategory', '')}",
            cuisine_type=meal.get("strArea", ""),
            prep_time=0,
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
        """Filter and rank meals based on user preferences"""
        
        if not meals:
            return []
        
        # Remove duplicates based on title similarity
        unique_meals = []
        seen_titles = set()
        
        for meal in meals:
            title_lower = meal.title.lower()
            # Simple duplicate detection
            is_duplicate = any(
                self._calculate_similarity(title_lower, seen_title) > 0.8 
                for seen_title in seen_titles
            )
            
            if not is_duplicate:
                unique_meals.append(meal)
                seen_titles.add(title_lower)
        
        # Score meals based on user preferences
        scored_meals = []
        for meal in unique_meals:
            score = await self._calculate_meal_score(meal, request_analysis)
            scored_meals.append((score, meal))
        
        # Sort by score and return top results
        scored_meals.sort(key=lambda x: x[0], reverse=True)
        return [meal for score, meal in scored_meals[:max_results]]
    
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
        """Calculate relevance score for a meal"""
        
        score = 0.0
        
        # Time preferences
        total_time = meal.prep_time + meal.cook_time
        max_time = request_analysis.get("max_cook_time", 60)
        if total_time <= max_time:
            score += 2.0
        elif total_time <= max_time * 1.5:
            score += 1.0
        
        # Dietary restrictions match
        meal_diets = [d.lower() for d in meal.diet_labels]
        user_diets = [d.lower() for d in request_analysis.get("dietary_restrictions", [])]
        for diet in user_diets:
            if any(diet in meal_diet for meal_diet in meal_diets):
                score += 2.0
        
        # Cuisine preference
        user_cuisines = [c.lower() for c in request_analysis.get("cuisine_types", [])]
        meal_cuisine = meal.cuisine_type.lower()
        for cuisine in user_cuisines:
            if cuisine in meal_cuisine:
                score += 1.5
        
        # Ingredient preferences
        meal_ingredients = [ing["name"].lower() for ing in meal.ingredients]
        
        # Bonus for included ingredients
        for ingredient in request_analysis.get("included_ingredients", []):
            if any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score += 1.0
        
        # Penalty for excluded ingredients
        for ingredient in request_analysis.get("excluded_ingredients", []):
            if any(ingredient.lower() in meal_ing for meal_ing in meal_ingredients):
                score -= 2.0
        
        # Servings match
        target_servings = request_analysis.get("servings", 2)
        servings_diff = abs(meal.servings - target_servings)
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
            "butter": "maslo"
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
                "summary": f"No meals found for '{original_request}'",
                "suggestions": [
                    "Try broader search terms",
                    "Check different cuisine types",
                    "Adjust dietary restrictions"
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
            "categories": self._categorize_meals(meals),
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
        
        Keep it friendly and helpful.
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
    
    def _categorize_meals(self, meals: List[MealResult]) -> Dict[str, List[str]]:
        """Categorize meals by various criteria"""
        
        categories = {
            "by_cuisine": {},
            "by_time": {"quick": [], "medium": [], "long": []},
            "by_diet": {},
            "by_difficulty": {"easy": [], "medium": [], "hard": []}
        }
        
        for meal in meals:
            # By cuisine
            cuisine = meal.cuisine_type or "international"
            if cuisine not in categories["by_cuisine"]:
                categories["by_cuisine"][cuisine] = []
            categories["by_cuisine"][cuisine].append(meal.title)
            
            # By time
            total_time = meal.prep_time + meal.cook_time
            if total_time <= 30:
                categories["by_time"]["quick"].append(meal.title)
            elif total_time <= 60:
                categories["by_time"]["medium"].append(meal.title)
            else:
                categories["by_time"]["long"].append(meal.title)
            
            # By diet
            for diet in meal.diet_labels:
                if diet not in categories["by_diet"]:
                    categories["by_diet"][diet] = []
                categories["by_diet"][diet].append(meal.title)
            
            # By difficulty
            difficulty = meal.difficulty.lower()
            if difficulty in categories["by_difficulty"]:
                categories["by_difficulty"][difficulty].append(meal.title)
        
        return categories
    
    def _generate_recommendations(
        self, 
        meals: List[MealResult], 
        request_analysis: Dict
    ) -> List[str]:
        """Generate helpful recommendations based on search results"""
        
        recommendations = []
        
        if not meals:
            recommendations.extend([
                "Try using broader search terms",
                "Consider different cuisine types",
                "Adjust your dietary restrictions if possible"
            ])
            return recommendations
        
        # Time-based recommendations
        quick_meals = [m for m in meals if (m.prep_time + m.cook_time) <= 30]
        if quick_meals and request_analysis.get("max_cook_time", 60) > 30:
            recommendations.append(f"For quick options, try: {quick_meals[0].title}")
        
        # Diet-specific recommendations
        diet_meals = [m for m in meals if any(
            diet in m.diet_labels 
            for diet in request_analysis.get("dietary_restrictions", [])
        )]
        if diet_meals:
            recommendations.append(f"Perfect for your diet: {diet_meals[0].title}")
        
        # Budget recommendations
        if any(meal.estimated_cost for meal in meals if meal.estimated_cost):
            cheapest = min(
                (m for m in meals if m.estimated_cost), 
                key=lambda x: x.estimated_cost
            )
            recommendations.append(f"Most budget-friendly: {cheapest.title} (€{cheapest.estimated_cost:.2f})")
        
        # Grocery integration recommendations
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

# Test function
async def test_meal_search():
    """Test the meal search functionality"""
    test_requests = [
        "Quick vegetarian dinner for 4 people",
        "Healthy breakfast options",
        "Italian pasta dishes under 30 minutes",
        "Keto-friendly lunch ideas",
        "Comfort food for family dinner"
    ]
    
    for request in test_requests:
        print(f"\n🔍 Testing: {request}")
        result = await search_meals_for_user(request, max_results=3)
        
        if result["success"]:
            print(f"✅ Found {result['filtered_count']} meals")
            print(f"📝 {result['presentation']['summary']}")
        else:
            print("❌ No meals found")

if __name__ == "__main__":
    asyncio.run(test_meal_search())