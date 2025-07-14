#!/usr/bin/env python3
"""
Fixed Meal Search Module with Proper Dietary Restriction Filtering
Addresses issues with vegetarian/vegan meal filtering and API parameter handling

MAJOR FIXES:
1. ✅ Fixed TheMealDB category filtering (was incorrectly skipped for dietary restrictions)
2. ✅ Added adaptive resource allocation (APIs get more quota when others fail)
3. ✅ Improved processing efficiency (processes more meals from successful APIs)
4. ✅ Enhanced logging to show allocation decisions and processing details
5. ✅ Fixed edge case where 40 meals found but only 4 processed (now processes up to 20)

KEY IMPROVEMENTS:
- Adaptive allocation: When APIs fail, working APIs get larger quotas
- Better TheMealDB utilization: Properly uses category and area filtering  
- Enhanced dietary filtering: Multi-stage filtering with comprehensive keyword lists
- Improved logging: Shows exact allocation and processing decisions
- Increased max results: Now returns up to 20 meals instead of 12
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
    Enhanced Meal Search with proper dietary restriction filtering
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
        
        # Dietary restriction keywords for filtering
        self.dietary_keywords = {
            "vegetarian": {
                "exclude": ["chicken", "beef", "pork", "lamb", "meat", "fish", "seafood", "bacon", "ham", "sausage", "turkey", "duck", "salmon", "tuna", "shrimp", "crab", "lobster"],
                "include": []
            },
            "vegan": {
                "exclude": ["chicken", "beef", "pork", "lamb", "meat", "fish", "seafood", "bacon", "ham", "sausage", "turkey", "duck", "salmon", "tuna", "shrimp", "crab", "lobster", "cheese", "milk", "butter", "cream", "egg", "yogurt", "honey"],
                "include": []
            },
            "gluten-free": {
                "exclude": ["wheat", "barley", "rye", "flour", "bread", "pasta", "noodles"],
                "include": []
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
        max_results: int = 20,
        include_grocery_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced meal search with proper dietary filtering
        """
        logger.info(f"🍽️ Searching meals for: '{user_request}'")
        
        try:
            # Step 1: Interpret the meal request
            request_analysis = await self._analyze_meal_request(user_request)
            logger.info(f"📋 Request analysis: {request_analysis}")
            
            # Step 2: Search meals across APIs with adaptive allocation
            all_meals = []
            
            async with aiohttp.ClientSession() as session:
                # Calculate initial allocation
                active_apis = sum(1 for api, config in self.apis.items() if config["enabled"])
                initial_per_api = max_results // active_apis if active_apis > 0 else max_results
                
                logger.info(f"📊 Initial allocation: {initial_per_api} meals per API ({active_apis} active APIs)")
                
                # Search Spoonacular (best for dietary restrictions)
                spoon_meals = []
                if self.apis["spoonacular"]["enabled"]:
                    spoon_meals = await self._search_spoonacular(session, request_analysis, initial_per_api)
                    all_meals.extend(spoon_meals)
                    logger.info(f"🥄 Spoonacular returned {len(spoon_meals)} meals")
                
                # Search Edamam (if working)
                edamam_meals = []
                if self.apis["edamam"]["enabled"]:
                    edamam_meals = await self._search_edamam(session, request_analysis, initial_per_api)
                    all_meals.extend(edamam_meals)
                    logger.info(f"🥗 Edamam returned {len(edamam_meals)} meals")
                
                # Search TheMealDB with adaptive allocation
                themealdb_quota = initial_per_api
                
                # If other APIs returned few/no results, give more quota to TheMealDB
                total_from_others = len(spoon_meals) + len(edamam_meals)
                if total_from_others < max_results // 2:
                    themealdb_quota = max_results - total_from_others
                    logger.info(f"🔄 Boosting TheMealDB quota to {themealdb_quota} (others returned {total_from_others})")
                
                themealdb_meals = await self._search_themealdb(session, request_analysis, themealdb_quota)
                all_meals.extend(themealdb_meals)
                logger.info(f"🍽️ TheMealDB returned {len(themealdb_meals)} meals")
            
            logger.info(f"📊 Total meals before filtering: {len(all_meals)}")
            
            # Step 3: Filter meals by dietary restrictions FIRST
            filtered_meals = self._filter_by_dietary_restrictions(all_meals, request_analysis)
            logger.info(f"🥬 Meals after dietary filtering: {len(filtered_meals)}")
            
            # Step 4: Remove duplicates and rank remaining meals
            unique_meals = self._remove_duplicates(filtered_meals)
            ranked_meals = await self._rank_meals(unique_meals, request_analysis)
            
            # Step 5: Limit to max results
            final_meals = ranked_meals[:max_results]
            
            # Step 6: Add nutrition estimates
            for meal in final_meals:
                meal["estimated_nutrition"] = self._estimate_nutrition(meal)
                meal["dietary_compliance"] = self._check_dietary_compliance(meal, request_analysis)
            
            result = {
                "success": True,
                "meals": final_meals,
                "total_found": len(all_meals),
                "after_dietary_filter": len(filtered_meals),
                "final_count": len(final_meals),
                "request_analysis": request_analysis,
                "apis_used": [api for api, config in self.apis.items() if config["enabled"]],
                "dietary_filtering_applied": bool(request_analysis.get("dietary_restrictions")),
                "summary": self._generate_meal_search_summary(user_request, final_meals, request_analysis)
            }
            
            logger.info(f"✅ Final result: {len(final_meals)} meals matching criteria")
            return result
            
        except Exception as e:
            logger.error(f"❌ Meal search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "meals": [],
                "message": "Failed to search for meals"
            }
    
    def _filter_by_dietary_restrictions(self, meals: List[Dict], analysis: Dict) -> List[Dict]:
        """
        Filter meals based on dietary restrictions using ingredient analysis
        """
        dietary_restrictions = analysis.get("dietary_restrictions", [])
        if not dietary_restrictions:
            return meals
        
        logger.info(f"🥬 Filtering {len(meals)} meals for dietary restrictions: {dietary_restrictions}")
        
        filtered_meals = []
        
        for meal in meals:
            # Get all ingredients as lowercase strings
            ingredients = meal.get("ingredients", [])
            ingredient_text = ""
            
            for ing in ingredients:
                if isinstance(ing, dict):
                    ingredient_text += f" {ing.get('name', '')} {ing.get('original', '')}"
                else:
                    ingredient_text += f" {str(ing)}"
            
            # Also check title and description
            meal_text = f"{meal.get('title', '')} {meal.get('description', '')} {ingredient_text}".lower()
            
            # Check if meal meets dietary requirements
            meets_requirements = True
            
            for restriction in dietary_restrictions:
                if restriction.lower() in self.dietary_keywords:
                    exclude_keywords = self.dietary_keywords[restriction.lower()]["exclude"]
                    
                    # Check if any excluded ingredients are present
                    for exclude_word in exclude_keywords:
                        if exclude_word.lower() in meal_text:
                            logger.debug(f"❌ Meal '{meal.get('title', '')}' contains '{exclude_word}' (excluded for {restriction})")
                            meets_requirements = False
                            break
                    
                    if not meets_requirements:
                        break
            
            if meets_requirements:
                filtered_meals.append(meal)
            else:
                logger.debug(f"🚫 Filtered out: {meal.get('title', 'Unknown')} (doesn't meet {dietary_restrictions})")
        
        logger.info(f"✅ Dietary filtering: {len(filtered_meals)}/{len(meals)} meals passed")
        return filtered_meals
    
    def _remove_duplicates(self, meals: List[Dict]) -> List[Dict]:
        """Remove duplicate meals based on title similarity"""
        seen_titles = set()
        unique_meals = []
        
        for meal in meals:
            title = meal.get("title", "").lower().strip()
            # Create a simplified version for comparison
            simple_title = "".join(c for c in title if c.isalnum() or c.isspace()).strip()
            
            if simple_title not in seen_titles and simple_title:
                seen_titles.add(simple_title)
                unique_meals.append(meal)
        
        return unique_meals
    
    async def _rank_meals(self, meals: List[Dict], analysis: Dict) -> List[Dict]:
        """Rank meals based on relevance to user request"""
        if not meals:
            return []
        
        scored_meals = []
        
        for meal in meals:
            score = self._calculate_meal_score(meal, analysis)
            scored_meals.append((score, meal))
        
        # Sort by score descending
        scored_meals.sort(key=lambda x: x[0], reverse=True)
        return [meal for score, meal in scored_meals]
    
    def _calculate_meal_score(self, meal: Dict, analysis: Dict) -> float:
        """Calculate relevance score for a meal"""
        score = 0.0
        
        # Base score for having a meal
        score += 1.0
        
        # Time preferences
        total_time = (meal.get("prep_time", 0) or 0) + (meal.get("cook_time", 0) or 0)
        max_time = analysis.get("max_cook_time", 60)
        
        if total_time <= max_time:
            score += 2.0
        elif total_time <= max_time * 1.5:
            score += 1.0
        
        # Meal type match
        meal_type = analysis.get("meal_type", "").lower()
        meal_title = meal.get("title", "").lower()
        meal_desc = meal.get("description", "").lower()
        
        if meal_type and meal_type != "any":
            if meal_type in meal_title or meal_type in meal_desc:
                score += 1.5
        
        # Cuisine match
        cuisine_types = analysis.get("cuisine_types", [])
        meal_cuisine = meal.get("cuisine_type", "").lower()
        
        for cuisine in cuisine_types:
            if cuisine.lower() in meal_cuisine or cuisine.lower() in meal_title:
                score += 1.0
                break
        
        # Keyword matches
        search_keywords = analysis.get("search_keywords", [])
        for keyword in search_keywords:
            if keyword.lower() in meal_title or keyword.lower() in meal_desc:
                score += 0.5
        
        # Dietary compliance boost (already filtered, so all should comply)
        dietary_restrictions = analysis.get("dietary_restrictions", [])
        if dietary_restrictions:
            score += 1.0  # Boost for meeting dietary requirements
        
        # Source reliability boost
        if meal.get("source") == "Spoonacular":
            score += 0.3  # Spoonacular generally has better data
        
        return score
    
    def _check_dietary_compliance(self, meal: Dict, analysis: Dict) -> Dict[str, Any]:
        """Check and report dietary compliance for a meal"""
        dietary_restrictions = analysis.get("dietary_restrictions", [])
        if not dietary_restrictions:
            return {"compliant": True, "restrictions_checked": []}
        
        compliance_result = {
            "compliant": True,
            "restrictions_checked": dietary_restrictions,
            "warnings": []
        }
        
        # Get meal text for checking
        ingredients = meal.get("ingredients", [])
        ingredient_text = ""
        
        for ing in ingredients:
            if isinstance(ing, dict):
                ingredient_text += f" {ing.get('name', '')} {ing.get('original', '')}"
            else:
                ingredient_text += f" {str(ing)}"
        
        meal_text = f"{meal.get('title', '')} {meal.get('description', '')} {ingredient_text}".lower()
        
        # Check each restriction
        for restriction in dietary_restrictions:
            if restriction.lower() in self.dietary_keywords:
                exclude_keywords = self.dietary_keywords[restriction.lower()]["exclude"]
                
                for exclude_word in exclude_keywords:
                    if exclude_word.lower() in meal_text:
                        compliance_result["compliant"] = False
                        compliance_result["warnings"].append(f"May contain {exclude_word}")
        
        return compliance_result
    
    # API SEARCH METHODS (Enhanced)
    async def _search_spoonacular(self, session: aiohttp.ClientSession, analysis: Dict, max_results: int) -> List[Dict]:
        """Enhanced Spoonacular search with better dietary restriction handling"""
        if not self.apis["spoonacular"]["enabled"]:
            return []
        
        try:
            params = {
                "apiKey": self.apis["spoonacular"]["api_key"],
                "number": max_results,
                "addRecipeInformation": "true",
                "fillIngredients": "true"
            }
            
            # Build query
            query_parts = []
            if analysis.get("english_query"):
                query_parts.append(analysis["english_query"])
            
            # Add meal type to query
            meal_type = analysis.get("meal_type", "")
            if meal_type and meal_type != "any":
                query_parts.append(meal_type)
            
            if query_parts:
                params["query"] = " ".join(query_parts)
            
            # Add dietary restrictions - this is crucial!
            dietary_restrictions = analysis.get("dietary_restrictions", [])
            if dietary_restrictions:
                # Spoonacular diet parameter mapping
                diet_map = {
                    "vegetarian": "vegetarian",
                    "vegan": "vegan", 
                    "gluten-free": "gluten free",
                    "ketogenic": "ketogenic",
                    "paleo": "paleo"
                }
                
                for diet in dietary_restrictions:
                    diet_lower = diet.lower()
                    if diet_lower in diet_map:
                        params["diet"] = diet_map[diet_lower]
                        logger.info(f"🥄 Spoonacular diet filter: {diet_map[diet_lower]}")
                        break
                
                # Also add as intolerances for stricter filtering
                intolerance_map = {
                    "gluten-free": "gluten"
                }
                
                intolerances = []
                for diet in dietary_restrictions:
                    if diet.lower() in intolerance_map:
                        intolerances.append(intolerance_map[diet.lower()])
                
                if intolerances:
                    params["intolerances"] = ",".join(intolerances)
            
            # Add time constraint
            if analysis.get("max_cook_time"):
                params["maxReadyTime"] = analysis["max_cook_time"]
            
            # Add cuisine filter
            cuisine_types = analysis.get("cuisine_types", [])
            if cuisine_types:
                params["cuisine"] = cuisine_types[0]  # Spoonacular takes one cuisine
            
            logger.info(f"🥄 Spoonacular params: {params}")
            
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
        """Enhanced Edamam search with better error handling"""
        if not self.apis["edamam"]["enabled"]:
            logger.info("🥗 Edamam disabled - no API credentials")
            return []
        
        try:
            # Build query
            query_parts = []
            if analysis.get("english_query"):
                query_parts.append(analysis["english_query"])
            
            # Add meal type
            meal_type = analysis.get("meal_type", "")
            if meal_type and meal_type != "any":
                query_parts.append(meal_type)
            
            params = {
                "type": "public",
                "app_id": self.apis["edamam"]["app_id"],
                "app_key": self.apis["edamam"]["app_key"],
                "to": max_results,
                "q": " ".join(query_parts) if query_parts else "healthy meal"
            }
            
            # Add dietary restrictions for Edamam
            dietary_restrictions = analysis.get("dietary_restrictions", [])
            if dietary_restrictions:
                # Edamam health labels
                health_map = {
                    "vegetarian": "vegetarian",
                    "vegan": "vegan",
                    "gluten-free": "gluten-free",
                    "dairy-free": "dairy-free"
                }
                
                health_labels = []
                for diet in dietary_restrictions:
                    if diet.lower() in health_map:
                        health_labels.append(health_map[diet.lower()])
                
                if health_labels:
                    params["health"] = health_labels[0]  # Edamam takes one health label
                    logger.info(f"🥗 Edamam health filter: {health_labels[0]}")
            
            headers = {
                "Accept": "application/json",
                "User-Agent": "Slovenian-Grocery-Intelligence/1.0"
            }
            
            logger.info(f"🥗 Edamam params: {params}")
            
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
                elif response.status == 401:
                    logger.error("🥗 Edamam authentication failed - check API credentials")
                    return []
                else:
                    logger.warning(f"Edamam API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Edamam search failed: {e}")
            return []
    
    async def _search_themealdb(self, session: aiohttp.ClientSession, analysis: Dict, max_results: int) -> List[Dict]:
        """
        Enhanced TheMealDB search with proper category and area filtering
        """
        try:
            meals = []
            dietary_restrictions = analysis.get("dietary_restrictions", [])
            cuisine_types = analysis.get("cuisine_types", [])
            
            logger.info(f"🍽️ TheMealDB search - dietary: {dietary_restrictions}, cuisine: {cuisine_types}")
            
            # Step 1: Filter by dietary category if specified
            if dietary_restrictions:
                # TheMealDB category mapping (based on their actual categories)
                category_map = {
                    "vegetarian": "Vegetarian",
                    "vegan": "Vegan",
                    "seafood": "Seafood",
                    "dessert": "Dessert",
                    "starter": "Starter"
                }
                
                for diet in dietary_restrictions:
                    diet_lower = diet.lower()
                    if diet_lower in category_map:
                        category = category_map[diet_lower]
                        logger.info(f"🥬 Searching TheMealDB for category: {category}")
                        
                        try:
                            async with session.get(f"{self.apis['themealdb']['base_url']}/filter.php?c={category}") as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data and data.get("meals"):
                                        logger.info(f"📊 TheMealDB found {len(data['meals'])} {category} meals")
                                        
                                        # Get detailed info for meals (limit to max_results to avoid too many API calls)
                                        meals_to_process = min(len(data["meals"]), max_results)
                                        logger.info(f"📋 Processing {meals_to_process} out of {len(data['meals'])} {category} meals")
                                        
                                        for meal_basic in data["meals"][:meals_to_process]:
                                            detailed_meal = await self._get_themealdb_details(session, meal_basic["idMeal"])
                                            if detailed_meal:
                                                meals.append(detailed_meal)
                                        
                                        if meals:
                                            logger.info(f"✅ Successfully retrieved {len(meals)} detailed {category} meals")
                                            return meals[:max_results]  # Return early if we found dietary-specific meals
                                else:
                                    logger.warning(f"TheMealDB category search failed with status: {response.status}")
                        except Exception as e:
                            logger.warning(f"TheMealDB category search failed: {e}")
            
            # Step 2: If no dietary results or no dietary restrictions, try cuisine filtering
            if len(meals) < max_results and cuisine_types:
                # TheMealDB area mapping
                area_map = {
                    "italian": "Italian",
                    "chinese": "Chinese", 
                    "mexican": "Mexican",
                    "indian": "Indian",
                    "mediterranean": "Greek",  # TheMealDB uses Greek for Mediterranean
                    "asian": "Thai",  # Use Thai as representative Asian cuisine
                    "american": "American",
                    "british": "British",
                    "french": "French"
                }
                
                for cuisine in cuisine_types[:2]:  # Try up to 2 cuisines
                    cuisine_lower = cuisine.lower()
                    if cuisine_lower in area_map:
                        area = area_map[cuisine_lower]
                        logger.info(f"🌍 Searching TheMealDB for area: {area}")
                        
                        try:
                            async with session.get(f"{self.apis['themealdb']['base_url']}/filter.php?a={area}") as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data and data.get("meals"):
                                        logger.info(f"📊 TheMealDB found {len(data['meals'])} {area} meals")
                                        
                                        # Get detailed info for a subset (more when this is our main source)
                                        meals_needed = max_results // 2 if dietary_restrictions else max_results // 3
                                        for meal_basic in data["meals"][:meals_needed]:
                                            detailed_meal = await self._get_themealdb_details(session, meal_basic["idMeal"])
                                            if detailed_meal:
                                                meals.append(detailed_meal)
                        except Exception as e:
                            logger.warning(f"TheMealDB area search failed: {e}")
            
            # Step 3: If still not enough results and no strict dietary restrictions, get some random meals
            if len(meals) < max_results and not dietary_restrictions:
                remaining_needed = max_results - len(meals)
                logger.info(f"🎲 Getting {remaining_needed} random TheMealDB meals")
                
                for _ in range(min(remaining_needed, 5)):  # Limit random requests
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
            
            logger.info(f"🍽️ TheMealDB returning {len(meals)} meals")
            return meals[:max_results]
            
        except Exception as e:
            logger.error(f"TheMealDB search failed: {e}")
            return []
    
    # ... (keep all the existing parsing methods and other helper methods unchanged)
    
    async def _analyze_meal_request(self, user_request: str) -> Dict[str, Any]:
        """Enhanced meal request analysis with better dietary extraction"""
        prompt = f"""
        Analyze this meal request: "{user_request}"
        
        Extract:
        1. Meal type (breakfast, lunch, dinner, snack, any)
        2. Cuisine preferences (italian, chinese, mexican, etc.)
        3. Dietary restrictions (vegetarian, vegan, gluten-free, keto, etc.) - BE VERY SPECIFIC
        4. Cooking time preferences (quick <30min, medium 30-60min, elaborate >60min)
        5. Number of servings needed
        6. Specific ingredients mentioned
        7. Health preferences (healthy, comfort food, balanced)
        8. Difficulty level (easy, medium, hard)
        
        IMPORTANT: If the user mentions "vegetarian", "vegan", "gluten-free" or any dietary restriction,
        make sure to include it in the dietary_restrictions array.
        
        Respond with JSON:
        {{
            "meal_type": "lunch",
            "cuisine_types": ["italian"],
            "dietary_restrictions": ["vegetarian"],
            "max_cook_time": 60,
            "servings": 4,
            "included_ingredients": ["chicken", "rice"],
            "excluded_ingredients": ["nuts"],
            "health_focus": "healthy",
            "difficulty": "medium",
            "search_keywords": ["vegetarian", "lunch", "healthy"],
            "english_query": "healthy vegetarian lunch recipes"
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
            
            # Ensure dietary restrictions are properly extracted
            if not analysis.get("dietary_restrictions"):
                # Fallback pattern matching
                user_lower = user_request.lower()
                if "vegetarian" in user_lower and "vegan" not in user_lower:
                    analysis["dietary_restrictions"] = ["vegetarian"]
                elif "vegan" in user_lower:
                    analysis["dietary_restrictions"] = ["vegan"]
                elif "gluten-free" in user_lower or "gluten free" in user_lower:
                    analysis["dietary_restrictions"] = ["gluten-free"]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Meal request analysis failed: {e}")
            # Fallback with basic pattern matching
            user_lower = user_request.lower()
            dietary_restrictions = []
            
            if "vegetarian" in user_lower and "vegan" not in user_lower:
                dietary_restrictions.append("vegetarian")
            elif "vegan" in user_lower:
                dietary_restrictions.append("vegan")
            
            if "gluten-free" in user_lower or "gluten free" in user_lower:
                dietary_restrictions.append("gluten-free")
            
            return {
                "meal_type": "lunch" if "lunch" in user_lower else "dinner" if "dinner" in user_lower else "any",
                "cuisine_types": [],
                "dietary_restrictions": dietary_restrictions,
                "max_cook_time": 60,
                "servings": 2,
                "included_ingredients": [],
                "excluded_ingredients": [],
                "health_focus": "balanced",
                "difficulty": "medium",
                "search_keywords": [user_request],
                "english_query": user_request
            }
    
    # Keep all existing parsing and helper methods...
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
    
    async def _get_themealdb_categories(self, session: aiohttp.ClientSession) -> List[str]:
        """Get all available categories from TheMealDB for future reference"""
        try:
            async with session.get(f"{self.apis['themealdb']['base_url']}/categories.php") as response:
                if response.status == 200:
                    data = await response.json()
                    if data and data.get("categories"):
                        categories = [cat["strCategory"] for cat in data["categories"]]
                        logger.info(f"📋 TheMealDB available categories: {categories}")
                        return categories
            return []
        except Exception as e:
            logger.warning(f"Failed to get TheMealDB categories: {e}")
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
    
    def _estimate_nutrition(self, meal: Dict) -> Dict[str, Any]:
        """Estimate basic nutrition for a meal"""
        ingredient_count = len(meal.get("ingredients", []))
        servings = meal.get("servings", 2)
        
        base_calories = 200 + (ingredient_count * 50)
        calories_per_serving = base_calories / servings if servings > 0 else base_calories
        
        return {
            "calories_per_serving": round(calories_per_serving),
            "estimated_protein": "15-25g",
            "estimated_carbs": "30-50g",
            "estimated_fat": "10-20g",
            "confidence": "estimated"
        }
    
    def _generate_meal_search_summary(self, user_request: str, meals: List[Dict], analysis: Dict) -> str:
        """Generate summary of meal search results"""
        if not meals:
            return f"No meals found for '{user_request}'. Try different search terms or relax dietary restrictions."
        
        total = len(meals)
        dietary_restrictions = analysis.get("dietary_restrictions", [])
        
        summary = f"Found {total} meal options for '{user_request}'"
        
        if dietary_restrictions:
            summary += f" matching {', '.join(dietary_restrictions)} dietary requirements"
        
        cuisines = len(set(meal.get("cuisine_type", "").split(",")[0] for meal in meals if meal.get("cuisine_type")))
        if cuisines > 1:
            summary += f" across {cuisines} cuisine types"
        
        summary += "."
        
        return summary

# Global meal searcher instance  
meal_searcher = MealSearcher()

async def search_meals(user_request: str, max_results: int = 20) -> Dict[str, Any]:
    """Main function to search meals with proper dietary filtering"""
    return await meal_searcher.search_meals(user_request, max_results, include_grocery_analysis=False)

# Keep all other existing functions unchanged...
async def get_meal_with_grocery_analysis(meal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to get meal with grocery analysis"""
    return await meal_searcher.get_meal_with_grocery_analysis(meal_data)

async def reverse_meal_search(available_ingredients: List[str], max_results: int = 10) -> Dict[str, Any]:
    """Main function for reverse meal search"""
    return await meal_searcher.reverse_meal_search(available_ingredients, max_results)