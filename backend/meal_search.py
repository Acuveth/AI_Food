#!/usr/bin/env python3
"""
Enhanced Meal Search Module with Slovenian Language Support
Addresses issues with vegetarian/vegan meal filtering and API parameter handling
Enhanced with Slovenian dietary terms and ingredient processing
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
    Enhanced Meal Search with Slovenian language support
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
        
        # Enhanced dietary restriction keywords for filtering - SLOVENIAN + ENGLISH
        self.dietary_keywords = {
            "vegetarian": {
                "slovenian": ["vegetarijansko", "vegetarijan", "brez mesa", "rastlinsko"],
                "english": ["vegetarian", "veggie", "plant-based"],
                "exclude": [
                    # English
                    "chicken", "beef", "pork", "lamb", "meat", "fish", "seafood", 
                    "bacon", "ham", "sausage", "turkey", "duck", "salmon", "tuna", 
                    "shrimp", "crab", "lobster",
                    # Slovenian
                    "piščanec", "piščančje", "goveje", "govedina", "svinjina", 
                    "svinjsko", "jagnjičje", "meso", "mesni", "riba", "ribje", 
                    "morski sadeži", "šunka", "klobasa", "puran", "račka", 
                    "losos", "tuna", "rake", "školjke"
                ]
            },
            "vegan": {
                "slovenian": ["veganski", "vegan", "rastlinsko", "brez živalskih"],
                "english": ["vegan", "plant-based", "dairy-free"],
                "exclude": [
                    # English
                    "chicken", "beef", "pork", "lamb", "meat", "fish", "seafood", 
                    "bacon", "ham", "sausage", "turkey", "duck", "salmon", "tuna", 
                    "shrimp", "crab", "lobster", "cheese", "milk", "butter", 
                    "cream", "egg", "yogurt", "honey",
                    # Slovenian
                    "piščanec", "piščančje", "goveje", "govedina", "svinjina", 
                    "svinjsko", "jagnjičje", "meso", "mesni", "riba", "ribje", 
                    "morski sadeži", "šunka", "klobasa", "puran", "račka", 
                    "losos", "tuna", "rake", "školjke", "sir", "mleko", "maslo", 
                    "smetana", "jajca", "jajce", "jogurt", "med"
                ]
            },
            "gluten-free": {
                "slovenian": ["brez glutena", "bezglutenske", "brez pšenice"],
                "english": ["gluten-free", "gluten free", "wheat-free"],
                "exclude": [
                    # English
                    "wheat", "barley", "rye", "flour", "bread", "pasta", "noodles",
                    # Slovenian
                    "pšenica", "pšenična", "ječmen", "ječmena", "rž", "ržena", 
                    "moka", "kruh", "testenine", "rezanci"
                ]
            },
            "healthy": {
                "slovenian": ["zdravo", "zdrava", "nizka maščoba", "organski", "bio"],
                "english": ["healthy", "low-fat", "organic", "natural"],
                "exclude": [
                    # English
                    "fried", "processed", "artificial", "junk",
                    # Slovenian
                    "cvrt", "ocvrt", "procesiran", "umetno", "nezdravo"
                ]
            }
        }
        
        logger.info(f"🔧 Meal APIs: Spoonacular={self.apis['spoonacular']['enabled']}, "
                   f"Edamam={self.apis['edamam']['enabled']}, TheMealDB={self.apis['themealdb']['enabled']}")
    
    async def _ensure_db_connection(self):
        """Ensure database connection is available"""
        if self.db_handler is None:
            self.db_handler = await get_db_handler()
    
    async def get_meal_with_grocery_analysis(self, meal_data: Dict[str, Any]) -> Dict[str, Any]:
            """Get detailed grocery cost analysis for a specific meal with Slovenian support"""
            await self._ensure_db_connection()
            
            logger.info(f"🛒 Analiza stroškov nakupovanja za jed: {meal_data.get('title', 'Unknown')}")
            
            try:
                # Extract ingredients from meal data
                ingredients = meal_data.get('ingredients', [])
                if not ingredients:
                    return {
                        "success": False,
                        "message": "V podatkih o jedi ni najdenih sestavin",
                        "meal": meal_data
                    }
                
                # Extract ingredient names for database search
                ingredient_names = []
                for ingredient in ingredients:
                    if isinstance(ingredient, dict):
                        name = ingredient.get('name', '') or ingredient.get('original', '')
                        if name:
                            ingredient_names.append(name)
                    elif isinstance(ingredient, str):
                        ingredient_names.append(ingredient)
                
                if not ingredient_names:
                    return {
                        "success": False,
                        "message": "Ni bilo mogoče ekstraktirati imen sestavin iz podatkov o jedi",
                        "meal": meal_data
                    }
                
                logger.info(f"🔍 Iščem {len(ingredient_names)} sestavin: {ingredient_names[:3]}...")
                
                # Find grocery prices for all ingredients
                ingredient_results = await self.db_handler.find_meal_ingredients(ingredient_names)
                
                # Analyze prices by store
                store_analysis = {}
                stores = ["dm", "lidl", "mercator", "spar", "tus"]
                
                for store in stores:
                    store_analysis[store] = {
                        "store_name": store.upper(),
                        "total_cost": 0,
                        "available_items": 0,
                        "missing_items": [],
                        "found_products": [],
                        "completeness": 0
                    }
                
                # Process each ingredient's results
                combined_cost = 0
                combined_items = []
                total_ingredients = len(ingredient_names)
                
                for ingredient, products in ingredient_results.items():
                    best_product = None
                    best_price = float('inf')
                    
                    # Find the cheapest option for this ingredient
                    for product in products:
                        current_price = product.get('current_price', 0) or 0
                        if current_price > 0 and current_price < best_price:
                            best_price = current_price
                            best_product = product
                    
                    if best_product:
                        combined_cost += best_price
                        combined_items.append({
                            "ingredient": ingredient,
                            "product": best_product,
                            "price": best_price,
                            "store": best_product.get('store_name', ''),
                            "found": True
                        })
                        
                        # Update store analysis
                        store_name = best_product.get('store_name', '').lower()
                        if store_name in store_analysis:
                            store_analysis[store_name]["available_items"] += 1
                            store_analysis[store_name]["total_cost"] += best_price
                            store_analysis[store_name]["found_products"].append(best_product)
                    else:
                        combined_items.append({
                            "ingredient": ingredient,
                            "product": None,
                            "price": 0,
                            "store": None,
                            "found": False
                        })
                        
                        # Add to missing items for all stores
                        for store_data in store_analysis.values():
                            store_data["missing_items"].append(ingredient)
                
                # Calculate completeness percentages
                for store_data in store_analysis.values():
                    if total_ingredients > 0:
                        store_data["completeness"] = (store_data["available_items"] / total_ingredients) * 100
                
                # Create combined analysis
                combined_analysis = {
                    "total_cost": round(combined_cost, 2),
                    "items_found": sum(1 for item in combined_items if item["found"]),
                    "items_missing": sum(1 for item in combined_items if not item["found"]),
                    "completeness": (sum(1 for item in combined_items if item["found"]) / total_ingredients * 100) if total_ingredients > 0 else 0,
                    "item_details": combined_items
                }
                
                # Calculate meal statistics
                servings = meal_data.get('servings', 2) or 2
                meal_statistics = {
                    "total_ingredients": total_ingredients,
                    "ingredients_found": combined_analysis["items_found"],
                    "cost_per_serving": round(combined_cost / servings, 2) if servings > 0 else 0,
                    "estimated_total": round(combined_cost, 2)
                }
                
                # Generate summary in Slovenian
                found_percentage = combined_analysis["completeness"]
                if found_percentage >= 80:
                    summary = f"Najdene cene živil za {meal_statistics['ingredients_found']}/{total_ingredients} sestavin. Ocenjeni skupni strošek: €{combined_analysis['total_cost']:.2f} (€{meal_statistics['cost_per_serving']:.2f} na porcijo)."
                elif found_percentage >= 50:
                    summary = f"Najdene cene za {meal_statistics['ingredients_found']}/{total_ingredients} sestavin. Nekatere izdelke boste morda morali kupiti drugje. Ocenjeni strošek: €{combined_analysis['total_cost']:.2f}."
                else:
                    summary = f"Najdene cene samo za {meal_statistics['ingredients_found']}/{total_ingredients} sestavin. Mnoge izdelke boste morda morali kupiti drugje ali nadomestiti."
                
                result = {
                    "success": True,
                    "meal": meal_data,
                    "grocery_analysis": {
                        "ingredient_results": ingredient_results,
                        "store_analysis": store_analysis,
                        "combined_analysis": combined_analysis,
                        "meal_statistics": meal_statistics
                    },
                    "summary": summary
                }
                
                logger.info(f"✅ Analiza nakupovanja dokončana: {meal_statistics['ingredients_found']}/{total_ingredients} sestavin najdenih, €{combined_analysis['total_cost']:.2f} skupni strošek")
                return result
                
            except Exception as e:
                logger.error(f"❌ Analiza nakupovanja ni uspela: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "meal": meal_data,
                    "message": "Analiza stroškov nakupovanja ni uspela"
                }
    
    # REVERSE MEAL SEARCH METHODS
    async def reverse_meal_search(self, available_ingredients: List[str], max_results: int = 10) -> Dict[str, Any]:
        """Find meals that can be made with available ingredients - with Slovenian support"""
        logger.info(f"🔄 Reverse meal search za sestavine: {available_ingredients}")
        
        try:
            # Build search query from ingredients
            ingredients_query = " ".join(available_ingredients)
            
            # Use existing meal search but with ingredient-focused query
            search_query = f"recepti z {ingredients_query}"
            
            # Search for meals
            meal_results = await self.search_meals(search_query, max_results)
            
            if not meal_results["success"]:
                return {
                    "success": False,
                    "message": "Iskanje jedi z vašimi sestavinami ni uspelo",
                    "available_ingredients": available_ingredients
                }
            
            # Score meals based on ingredient matches
            scored_meals = []
            for meal in meal_results["meals"]:
                match_score = self._calculate_ingredient_match_score(meal, available_ingredients)
                meal["ingredient_match_score"] = match_score
                scored_meals.append(meal)
            
            # Sort by match score
            scored_meals.sort(key=lambda x: x["ingredient_match_score"], reverse=True)
            
            # Generate summary in Slovenian
            summary = f"Najdenih {len(scored_meals)} jedi, ki jih lahko pripravite z vašimi sestavinami: {', '.join(available_ingredients)}"
            
            return {
                "success": True,
                "suggested_meals": scored_meals,
                "available_ingredients": available_ingredients,
                "total_found": len(scored_meals),
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"❌ Reverse meal search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "available_ingredients": available_ingredients,
                "message": "Iskanje jedi z vašimi sestavinami ni uspelo"
            }
    
    def _calculate_ingredient_match_score(self, meal: Dict, available_ingredients: List[str]) -> float:
        """Calculate how well a meal matches available ingredients"""
        meal_ingredients = meal.get("ingredients", [])
        if not meal_ingredients:
            return 0.0
        
        # Get ingredient names from meal
        meal_ingredient_names = []
        for ing in meal_ingredients:
            if isinstance(ing, dict):
                name = ing.get('name', '') or ing.get('original', '')
                if name:
                    meal_ingredient_names.append(name.lower())
            elif isinstance(ing, str):
                meal_ingredient_names.append(ing.lower())
        
        if not meal_ingredient_names:
            return 0.0
        
        # Calculate matches
        available_lower = [ing.lower() for ing in available_ingredients]
        matches = 0
        
        for meal_ing in meal_ingredient_names:
            for available_ing in available_lower:
                if available_ing in meal_ing or meal_ing in available_ing:
                    matches += 1
                    break
        
        # Calculate score as percentage
        match_score = (matches / len(meal_ingredient_names)) * 100
        return min(match_score, 100.0)

    def _parse_spoonacular_recipe(self, recipe: Dict) -> Optional[Dict]:
        """Parse Spoonacular recipe data with enhanced error handling"""
        try:
            ingredients = []
            for ing in recipe.get("extendedIngredients", []):
                ingredients.append({
                    "name": ing.get("name", ""),
                    "amount": ing.get("amount", 0),
                    "unit": ing.get("unit", ""),
                    "original": ing.get("original", "")
                })
            
            # Clean HTML from summary
            summary = recipe.get("summary", "")
            if summary:
                import re
                summary = re.sub(r'<[^>]+>', '', summary)[:200]
            
            return {
                "id": f"spoon_{recipe.get('id', '')}",
                "title": recipe.get("title", "Unknown Recipe"),
                "description": summary,
                "image_url": recipe.get("image", ""),
                "prep_time": recipe.get("preparationMinutes", 0) or 0,
                "cook_time": recipe.get("cookingMinutes", 0) or 0,
                "servings": recipe.get("servings", 2) or 2,
                "ingredients": ingredients,
                "instructions": self._parse_spoonacular_instructions(recipe.get("instructions", "")),
                "cuisine_type": ",".join(recipe.get("cuisines", [])),
                "diet_labels": recipe.get("diets", []),
                "recipe_url": recipe.get("sourceUrl", ""),
                "source": "Spoonacular",
                "difficulty": self._determine_difficulty(recipe),
                "nutrition": self._parse_spoonacular_nutrition(recipe.get("nutrition", {}))
            }
        except Exception as e:
            logger.warning(f"Error parsing Spoonacular recipe: {e}")
            return None
    
    def _parse_spoonacular_instructions(self, instructions: str) -> List[str]:
        """Parse Spoonacular instructions into steps"""
        if not instructions:
            return []
        
        # Split by periods and clean up
        steps = []
        for step in instructions.split('.'):
            step = step.strip()
            if step and len(step) > 10:  # Filter out very short steps
                steps.append(step)
        
        return steps
    
    def _parse_spoonacular_nutrition(self, nutrition: Dict) -> Dict[str, Any]:
        """Parse Spoonacular nutrition data"""
        if not nutrition:
            return {}
        
        nutrients = nutrition.get("nutrients", [])
        nutrition_data = {}
        
        for nutrient in nutrients:
            name = nutrient.get("name", "").lower()
            amount = nutrient.get("amount", 0)
            unit = nutrient.get("unit", "")
            
            if "calorie" in name:
                nutrition_data["calories"] = amount
            elif "protein" in name:
                nutrition_data["protein"] = f"{amount}{unit}"
            elif "carbohydrate" in name:
                nutrition_data["carbs"] = f"{amount}{unit}"
            elif "fat" in name:
                nutrition_data["fat"] = f"{amount}{unit}"
        
        return nutrition_data
    
    def _parse_edamam_recipe(self, recipe: Dict) -> Optional[Dict]:
        """Parse Edamam recipe data with enhanced error handling"""
        try:
            ingredients = []
            for ing in recipe.get("ingredients", []):
                ingredients.append({
                    "name": ing.get("food", ""),
                    "amount": ing.get("quantity", 0) or 0,
                    "unit": ing.get("measure", ""),
                    "original": ing.get("text", "")
                })
            
            # Get cuisine type
            cuisine_type = ""
            if recipe.get("cuisineType"):
                cuisine_type = recipe["cuisineType"][0] if isinstance(recipe["cuisineType"], list) else recipe["cuisineType"]
            
            return {
                "id": f"edamam_{recipe.get('uri', '').split('_')[-1] if recipe.get('uri') else 'unknown'}",
                "title": recipe.get("label", "Unknown Recipe"),
                "description": f"Okusna {cuisine_type} kuhinja" if cuisine_type else "Okusna jed",
                "image_url": recipe.get("image", ""),
                "prep_time": 0,
                "cook_time": recipe.get("totalTime", 30) or 30,
                "servings": recipe.get("yield", 2) or 2,
                "ingredients": ingredients,
                "instructions": [],  # Edamam doesn't provide instructions
                "cuisine_type": cuisine_type,
                "diet_labels": recipe.get("dietLabels", []),
                "recipe_url": recipe.get("url", ""),
                "source": "Edamam",
                "difficulty": "medium",
                "nutrition": self._parse_edamam_nutrition(recipe.get("totalNutrients", {}))
            }
        except Exception as e:
            logger.warning(f"Error parsing Edamam recipe: {e}")
            return None
    
    def _parse_edamam_nutrition(self, nutrients: Dict) -> Dict[str, Any]:
        """Parse Edamam nutrition data"""
        if not nutrients:
            return {}
        
        nutrition_data = {}
        
        if "ENERC_KCAL" in nutrients:
            nutrition_data["calories"] = nutrients["ENERC_KCAL"].get("quantity", 0)
        
        if "PROCNT" in nutrients:
            nutrition_data["protein"] = f"{nutrients['PROCNT'].get('quantity', 0):.1f}g"
        
        if "CHOCDF" in nutrients:
            nutrition_data["carbs"] = f"{nutrients['CHOCDF'].get('quantity', 0):.1f}g"
        
        if "FAT" in nutrients:
            nutrition_data["fat"] = f"{nutrients['FAT'].get('quantity', 0):.1f}g"
        
        return nutrition_data
    
    def _parse_themealdb_recipe(self, meal: Dict) -> Optional[Dict]:
        """Parse TheMealDB recipe data with enhanced error handling"""
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
                # Split by periods and clean up
                steps = instructions_text.split(".")
                for step in steps:
                    step = step.strip()
                    if step and len(step) > 10:  # Filter out very short steps
                        instructions.append(step)
            
            # Determine cuisine type
            area = meal.get("strArea", "International")
            cuisine_type = area.lower()
            
            return {
                "id": f"mealdb_{meal.get('idMeal', '')}",
                "title": meal.get("strMeal", "Unknown Recipe"),
                "description": f"Tradicionalna {area} {meal.get('strCategory', 'jed')}",
                "image_url": meal.get("strMealThumb", ""),
                "prep_time": 0,
                "cook_time": 30,
                "servings": 4,
                "ingredients": ingredients,
                "instructions": instructions,
                "cuisine_type": cuisine_type,
                "diet_labels": self._determine_themealdb_diet_labels(meal),
                "recipe_url": meal.get("strSource", ""),
                "source": "TheMealDB",
                "difficulty": "medium",
                "nutrition": {}  # TheMealDB doesn't provide nutrition
            }
        except Exception as e:
            logger.warning(f"Error parsing TheMealDB recipe: {e}")
            return None
    
    def _determine_themealdb_diet_labels(self, meal: Dict) -> List[str]:
        """Determine diet labels for TheMealDB meals"""
        diet_labels = []
        
        category = meal.get("strCategory", "").lower()
        meal_name = meal.get("strMeal", "").lower()
        instructions = meal.get("strInstructions", "").lower()
        
        # Get all ingredients
        ingredients_text = ""
        for i in range(1, 21):
            ingredient = meal.get(f"strIngredient{i}", "")
            if ingredient and ingredient.strip():
                ingredients_text += f" {ingredient.lower()}"
        
        full_text = f"{category} {meal_name} {instructions} {ingredients_text}"
        
        # Check for vegetarian (no meat)
        meat_keywords = ["chicken", "beef", "pork", "lamb", "fish", "seafood", "meat", "bacon", "ham"]
        if not any(keyword in full_text for keyword in meat_keywords):
            diet_labels.append("vegetarian")
        
        # Check for vegan (no animal products)
        animal_products = ["milk", "cheese", "butter", "cream", "egg", "honey", "yogurt"]
        if not any(keyword in full_text for keyword in meat_keywords + animal_products):
            diet_labels.append("vegan")
        
        # Check for specific categories
        if "vegetarian" in category:
            diet_labels.append("vegetarian")
        
        if "vegan" in category:
            diet_labels.append("vegan")
        
        return diet_labels
    
    def _determine_difficulty(self, recipe: Dict) -> str:
        """Determine recipe difficulty based on various factors"""
        # Default difficulty
        difficulty = "medium"
        
        # Check cooking time
        total_time = (recipe.get("preparationMinutes", 0) or 0) + (recipe.get("cookingMinutes", 0) or 0)
        ingredient_count = len(recipe.get("extendedIngredients", []))
        
        if total_time <= 30 and ingredient_count <= 5:
            difficulty = "easy"
        elif total_time > 90 or ingredient_count > 15:
            difficulty = "hard"
        
        # Check instructions complexity
        instructions = recipe.get("instructions", "")
        if instructions:
            complex_words = ["marinate", "reduce", "fold", "temper", "caramelize", "braise"]
            if any(word in instructions.lower() for word in complex_words):
                difficulty = "hard"
        
        return difficulty
    
    def _estimate_nutrition(self, meal: Dict) -> Dict[str, Any]:
        """Estimate basic nutrition for a meal"""
        # Return existing nutrition if available
        if meal.get("nutrition"):
            return meal["nutrition"]
        
        ingredient_count = len(meal.get("ingredients", []))
        servings = meal.get("servings", 2)
        
        # Base estimation
        base_calories = 200 + (ingredient_count * 50)
        calories_per_serving = base_calories / servings if servings > 0 else base_calories
        
        return {
            "calories": round(calories_per_serving),
            "protein": "15-25g",
            "carbs": "30-50g",
            "fat": "10-20g",
            "confidence": "estimated"
        }


    async def _search_spoonacular(self, session: aiohttp.ClientSession, analysis: Dict, max_results: int) -> List[Dict]:
            """Enhanced Spoonacular search with Slovenian dietary restriction handling"""
            if not self.apis["spoonacular"]["enabled"]:
                return []
            
            try:
                params = {
                    "apiKey": self.apis["spoonacular"]["api_key"],
                    "number": max_results,
                    "addRecipeInformation": "true",
                    "fillIngredients": "true"
                }
                
                # Build query with Slovenian support
                query_parts = []
                if analysis.get("english_query"):
                    query_parts.append(analysis["english_query"])
                
                # Add meal type to query with Slovenian conversion
                meal_type = analysis.get("meal_type", "")
                slovenian_to_english_meals = {
                    "zajtrk": "breakfast",
                    "kosilo": "lunch",
                    "večerja": "dinner",
                    "malica": "snack",
                    "sladica": "dessert"
                }
                
                if meal_type in slovenian_to_english_meals:
                    meal_type = slovenian_to_english_meals[meal_type]
                
                if meal_type and meal_type != "any":
                    query_parts.append(meal_type)
                
                if query_parts:
                    params["query"] = " ".join(query_parts)
                
                # Add dietary restrictions - crucial for Slovenian terms!
                dietary_restrictions = analysis.get("dietary_restrictions", [])
                if dietary_restrictions:
                    # Convert Slovenian dietary terms to Spoonacular format
                    diet_map = {
                        "vegetarian": "vegetarian",
                        "vegetarijansko": "vegetarian",
                        "vegan": "vegan",
                        "veganski": "vegan",
                        "gluten-free": "gluten free",
                        "brez glutena": "gluten free",
                        "ketogenic": "ketogenic",
                        "keto": "ketogenic",
                        "paleo": "paleo",
                        "healthy": "whole30",
                        "zdravo": "whole30"
                    }
                    
                    for diet in dietary_restrictions:
                        diet_lower = diet.lower()
                        if diet_lower in diet_map:
                            params["diet"] = diet_map[diet_lower]
                            logger.info(f"🥄 Spoonacular diet filter: {diet_map[diet_lower]}")
                            break
                    
                    # Also add as intolerances for stricter filtering
                    intolerance_map = {
                        "gluten-free": "gluten",
                        "brez glutena": "gluten",
                        "dairy-free": "dairy",
                        "brez mleka": "dairy"
                    }
                    
                    intolerances = []
                    for diet in dietary_restrictions:
                        diet_lower = diet.lower()
                        if diet_lower in intolerance_map:
                            intolerances.append(intolerance_map[diet_lower])
                    
                    if intolerances:
                        params["intolerances"] = ",".join(intolerances)
                
                # Add time constraint
                if analysis.get("max_cook_time"):
                    params["maxReadyTime"] = analysis["max_cook_time"]
                
                # Add cuisine filter with Slovenian conversion
                cuisine_types = analysis.get("cuisine_types", [])
                if cuisine_types:
                    cuisine_map = {
                        "italijanska": "italian",
                        "kitajska": "chinese",
                        "mehiška": "mexican",
                        "indijska": "indian",
                        "slovenska": "eastern european",
                        "mediteranska": "mediterranean",
                        "azijska": "asian"
                    }
                    
                    cuisine = cuisine_types[0].lower()
                    if cuisine in cuisine_map:
                        params["cuisine"] = cuisine_map[cuisine]
                    else:
                        params["cuisine"] = cuisine
                
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
        """Enhanced Edamam search with Slovenian support"""
        if not self.apis["edamam"]["enabled"]:
            logger.info("🥗 Edamam disabled - no API credentials")
            return []
        
        try:
            # Build query with Slovenian support
            query_parts = []
            if analysis.get("english_query"):
                query_parts.append(analysis["english_query"])
            
            # Add meal type with Slovenian conversion
            meal_type = analysis.get("meal_type", "")
            slovenian_to_english_meals = {
                "zajtrk": "breakfast",
                "kosilo": "lunch",
                "večerja": "dinner",
                "malica": "snack",
                "sladica": "dessert"
            }
            
            if meal_type in slovenian_to_english_meals:
                meal_type = slovenian_to_english_meals[meal_type]
            
            if meal_type and meal_type != "any":
                query_parts.append(meal_type)
            
            params = {
                "type": "public",
                "app_id": self.apis["edamam"]["app_id"],
                "app_key": self.apis["edamam"]["app_key"],
                "to": max_results,
                "q": " ".join(query_parts) if query_parts else "healthy meal"
            }
            
            # Add dietary restrictions for Edamam with Slovenian support
            dietary_restrictions = analysis.get("dietary_restrictions", [])
            if dietary_restrictions:
                # Edamam health labels with Slovenian mapping
                health_map = {
                    "vegetarian": "vegetarian",
                    "vegetarijansko": "vegetarian",
                    "vegan": "vegan",
                    "veganski": "vegan",
                    "gluten-free": "gluten-free",
                    "brez glutena": "gluten-free",
                    "dairy-free": "dairy-free",
                    "brez mleka": "dairy-free",
                    "healthy": "low-sodium",
                    "zdravo": "low-sodium"
                }
                
                health_labels = []
                for diet in dietary_restrictions:
                    diet_lower = diet.lower()
                    if diet_lower in health_map:
                        health_labels.append(health_map[diet_lower])
                
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
        """Enhanced TheMealDB search with Slovenian dietary category and area filtering"""
        try:
            meals = []
            dietary_restrictions = analysis.get("dietary_restrictions", [])
            cuisine_types = analysis.get("cuisine_types", [])
            
            logger.info(f"🍽️ TheMealDB search - dietary: {dietary_restrictions}, cuisine: {cuisine_types}")
            
            # Step 1: Filter by dietary category if specified (with Slovenian support)
            if dietary_restrictions:
                # TheMealDB category mapping with Slovenian terms
                category_map = {
                    "vegetarian": "Vegetarian",
                    "vegetarijansko": "Vegetarian",
                    "vegan": "Vegan",
                    "veganski": "Vegan",
                    "seafood": "Seafood",
                    "morski sadeži": "Seafood",
                    "dessert": "Dessert",
                    "sladica": "Dessert",
                    "starter": "Starter",
                    "predjed": "Starter"
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
                                        
                                        # Get detailed info for meals (limit to max_results)
                                        meals_to_process = min(len(data["meals"]), max_results)
                                        logger.info(f"📋 Processing {meals_to_process} out of {len(data['meals'])} {category} meals")
                                        
                                        for meal_basic in data["meals"][:meals_to_process]:
                                            detailed_meal = await self._get_themealdb_details(session, meal_basic["idMeal"])
                                            if detailed_meal:
                                                meals.append(detailed_meal)
                                        
                                        if meals:
                                            logger.info(f"✅ Successfully retrieved {len(meals)} detailed {category} meals")
                                            return meals[:max_results]
                                else:
                                    logger.warning(f"TheMealDB category search failed with status: {response.status}")
                        except Exception as e:
                            logger.warning(f"TheMealDB category search failed: {e}")
            
            # Step 2: Try cuisine filtering with Slovenian support
            if len(meals) < max_results and cuisine_types:
                # TheMealDB area mapping with Slovenian terms
                area_map = {
                    "italian": "Italian",
                    "italijanska": "Italian",
                    "chinese": "Chinese",
                    "kitajska": "Chinese",
                    "mexican": "Mexican",
                    "mehiška": "Mexican",
                    "indian": "Indian",
                    "indijska": "Indian",
                    "mediterranean": "Greek",
                    "mediteranska": "Greek",
                    "asian": "Thai",
                    "azijska": "Thai",
                    "american": "American",
                    "ameriška": "American",
                    "british": "British",
                    "britanska": "British",
                    "french": "French",
                    "francoska": "French",
                    "slovenian": "Croatian",  
                    "slovenska": "Croatian"
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
                                        
                                        # Get detailed info for a subset
                                        meals_needed = max_results // 2 if dietary_restrictions else max_results // 3
                                        for meal_basic in data["meals"][:meals_needed]:
                                            detailed_meal = await self._get_themealdb_details(session, meal_basic["idMeal"])
                                            if detailed_meal:
                                                meals.append(detailed_meal)
                        except Exception as e:
                            logger.warning(f"TheMealDB area search failed: {e}")
            
            # Step 3: Get random meals if not enough results and no strict dietary restrictions
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
    



    async def search_meals(
        self,
        user_request: str,
        max_results: int = 20,
        include_grocery_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced meal search with Slovenian language support
        """
        logger.info(f"🍽️ Searching meals for: '{user_request}'")
        
        try:
            # Step 1: Interpret the meal request with Slovenian support
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
            
            # Step 3: Filter meals by dietary restrictions FIRST (with Slovenian support)
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
                "message": "Iskanje jedi ni uspelo"
            }
    
    def _filter_by_dietary_restrictions(self, meals: List[Dict], analysis: Dict) -> List[Dict]:
        """
        Filter meals based on dietary restrictions using Slovenian + English ingredient analysis
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
                restriction_key = restriction.lower()
                
                # Handle both Slovenian and English terms
                if restriction_key in ["vegetarijansko", "vegetarian"]:
                    restriction_key = "vegetarian"
                elif restriction_key in ["veganski", "vegan"]:
                    restriction_key = "vegan"
                elif restriction_key in ["brez glutena", "gluten-free", "gluten free"]:
                    restriction_key = "gluten-free"
                elif restriction_key in ["zdravo", "healthy"]:
                    restriction_key = "healthy"
                
                if restriction_key in self.dietary_keywords:
                    exclude_keywords = self.dietary_keywords[restriction_key]["exclude"]
                    
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
        
        # Meal type match (support Slovenian terms)
        meal_type = analysis.get("meal_type", "").lower()
        meal_title = meal.get("title", "").lower()
        meal_desc = meal.get("description", "").lower()
        
        # Handle Slovenian meal types
        slovenian_meal_types = {
            "zajtrk": "breakfast",
            "kosilo": "lunch", 
            "večerja": "dinner",
            "malica": "snack",
            "sladica": "dessert"
        }
        
        if meal_type in slovenian_meal_types:
            meal_type = slovenian_meal_types[meal_type]
        
        if meal_type and meal_type != "any":
            if meal_type in meal_title or meal_type in meal_desc:
                score += 1.5
        
        # Cuisine match
        cuisine_types = analysis.get("cuisine_types", [])
        meal_cuisine = meal.get("cuisine_type", "").lower()
        
        for cuisine in cuisine_types:
            cuisine_lower = cuisine.lower()
            # Handle Slovenian cuisine names
            if cuisine_lower in ["italijanska", "italian"]:
                cuisine_lower = "italian"
            elif cuisine_lower in ["kitajska", "chinese"]:
                cuisine_lower = "chinese"
            elif cuisine_lower in ["mehiška", "mexican"]:
                cuisine_lower = "mexican"
            elif cuisine_lower in ["slovenska", "slovenian"]:
                cuisine_lower = "slovenian"
            
            if cuisine_lower in meal_cuisine or cuisine_lower in meal_title:
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
            restriction_key = restriction.lower()
            
            # Normalize Slovenian terms
            if restriction_key in ["vegetarijansko", "vegetarian"]:
                restriction_key = "vegetarian"
            elif restriction_key in ["veganski", "vegan"]:
                restriction_key = "vegan"
            elif restriction_key in ["brez glutena", "gluten-free"]:
                restriction_key = "gluten-free"
            elif restriction_key in ["zdravo", "healthy"]:
                restriction_key = "healthy"
            
            if restriction_key in self.dietary_keywords:
                exclude_keywords = self.dietary_keywords[restriction_key]["exclude"]
                
                for exclude_word in exclude_keywords:
                    if exclude_word.lower() in meal_text:
                        compliance_result["compliant"] = False
                        compliance_result["warnings"].append(f"Lahko vsebuje {exclude_word}")
        
        return compliance_result
    
    async def _analyze_meal_request(self, user_request: str) -> Dict[str, Any]:
        """Enhanced meal request analysis with Slovenian language support"""
        prompt = f"""
        Analiziraj to zahtevo za jed: "{user_request}"
        
        Uporabnik lahko piše v slovenščini ali angleščini. Analiziraj in ekstraktiraj:
        
        1. Vrsta obroka (zajtrk/breakfast, kosilo/lunch, večerja/dinner, malica/snack, any)
        2. Kulinarične preference (italijanska/italian, kitajska/chinese, mehiška/mexican, slovenska/slovenian, itd.)
        3. Prehranske omejitve (vegetarijansko/vegetarian, veganski/vegan, brez glutena/gluten-free, keto, zdravo/healthy, itd.)
        4. Preference za čas kuhanja (hitro <30min, srednje 30-60min, zahtevno >60min)
        5. Število porcij
        6. Specifične omenjene sestavine
        7. Zdravstvene preference (zdravo/healthy, udobna hrana/comfort food, uravnoteženo/balanced)
        8. Težavnost (enostavno/easy, srednje/medium, težko/hard)
        
        POMEMBNO: Če uporabnik omeni "vegetarijansko", "veganski", "brez glutena" ali katero koli prehransko omejitev,
        vključi jo v dietary_restrictions array.
        
        Odgovori z JSON:
        {{
            "meal_type": "kosilo",
            "cuisine_types": ["italijanska"],
            "dietary_restrictions": ["vegetarijansko"],
            "max_cook_time": 60,
            "servings": 4,
            "included_ingredients": ["piščanec", "riž"],
            "excluded_ingredients": ["orehi"],
            "health_focus": "zdravo",
            "difficulty": "srednje",
            "search_keywords": ["vegetarijansko", "kosilo", "zdravo"],
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
                    max_tokens=700
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
            
            analysis = json.loads(json_text)
            
            # Ensure dietary restrictions are properly extracted with Slovenian support
            if not analysis.get("dietary_restrictions"):
                # Fallback pattern matching for Slovenian terms
                user_lower = user_request.lower()
                dietary_restrictions = []
                
                if any(term in user_lower for term in ["vegetarijansko", "vegetarian"]) and not any(term in user_lower for term in ["veganski", "vegan"]):
                    dietary_restrictions.append("vegetarian")
                elif any(term in user_lower for term in ["veganski", "vegan"]):
                    dietary_restrictions.append("vegan")
                
                if any(term in user_lower for term in ["brez glutena", "gluten-free", "gluten free"]):
                    dietary_restrictions.append("gluten-free")
                
                if any(term in user_lower for term in ["zdravo", "healthy"]):
                    dietary_restrictions.append("healthy")
                
                analysis["dietary_restrictions"] = dietary_restrictions
            
            return analysis
            
        except Exception as e:
            logger.error(f"Meal request analysis failed: {e}")
            # Fallback with basic pattern matching
            user_lower = user_request.lower()
            dietary_restrictions = []
            
            if any(term in user_lower for term in ["vegetarijansko", "vegetarian"]) and not any(term in user_lower for term in ["veganski", "vegan"]):
                dietary_restrictions.append("vegetarian")
            elif any(term in user_lower for term in ["veganski", "vegan"]):
                dietary_restrictions.append("vegan")
            
            if any(term in user_lower for term in ["brez glutena", "gluten-free", "gluten free"]):
                dietary_restrictions.append("gluten-free")
            
            if any(term in user_lower for term in ["zdravo", "healthy"]):
                dietary_restrictions.append("healthy")
            
            # Determine meal type
            meal_type = "any"
            if any(term in user_lower for term in ["zajtrk", "breakfast"]):
                meal_type = "breakfast"
            elif any(term in user_lower for term in ["kosilo", "lunch"]):
                meal_type = "lunch"
            elif any(term in user_lower for term in ["večerja", "dinner"]):
                meal_type = "dinner"
            
            return {
                "meal_type": meal_type,
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
    
    # Keep all existing API search methods and helper functions...
    # (The existing methods remain unchanged)
    
    def _generate_meal_search_summary(self, user_request: str, meals: List[Dict], analysis: Dict) -> str:
        """Generate summary of meal search results in Slovenian"""
        if not meals:
            return f"Ni najdenih jedi za '{user_request}'. Poskusite z drugačnimi iskalnimi pojmi ali sprostite prehranske omejitve."
        
        total = len(meals)
        dietary_restrictions = analysis.get("dietary_restrictions", [])
        
        # Convert dietary restrictions to Slovenian for summary
        slovenian_restrictions = []
        for restriction in dietary_restrictions:
            if restriction == "vegetarian":
                slovenian_restrictions.append("vegetarijansko")
            elif restriction == "vegan":
                slovenian_restrictions.append("veganski")
            elif restriction == "gluten-free":
                slovenian_restrictions.append("brez glutena")
            elif restriction == "healthy":
                slovenian_restrictions.append("zdravo")
            else:
                slovenian_restrictions.append(restriction)
        
        summary = f"Najdenih {total} možnosti jedi za '{user_request}'"
        
        if slovenian_restrictions:
            summary += f" z upoštevanjem {', '.join(slovenian_restrictions)} prehranskih zahtev"
        
        cuisines = len(set(meal.get("cuisine_type", "").split(",")[0] for meal in meals if meal.get("cuisine_type")))
        if cuisines > 1:
            summary += f" iz {cuisines} različnih kuhinj"
        
        summary += "."
        
        return summary
    
    # [Keep all existing API search methods unchanged - _search_spoonacular, _search_edamam, _search_themealdb, etc.]
    # [Keep all existing parsing methods unchanged - _parse_spoonacular_recipe, _parse_edamam_recipe, etc.]
    # [Keep all existing helper methods unchanged - _estimate_nutrition, etc.]

# Global meal searcher instance  
meal_searcher = MealSearcher()

async def search_meals(user_request: str, max_results: int = 20) -> Dict[str, Any]:
    """Main function to search meals with Slovenian language support"""
    return await meal_searcher.search_meals(user_request, max_results, include_grocery_analysis=False)

async def get_meal_with_grocery_analysis(meal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to get meal with grocery analysis"""
    return await meal_searcher.get_meal_with_grocery_analysis(meal_data)

async def reverse_meal_search(available_ingredients: List[str], max_results: int = 10) -> Dict[str, Any]:
    """Main function for reverse meal search"""
    return await meal_searcher.reverse_meal_search(available_ingredients, max_results)