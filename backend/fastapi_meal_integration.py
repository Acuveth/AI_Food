#!/usr/bin/env python3
"""
FastAPI Integration for Enhanced Meal Search
Adds meal search endpoints to your existing grocery intelligence backend
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import asyncio
import logging
from typing import Dict

# Import your meal search system
from meal_search import EnhancedMealSearchManager, search_meals_for_user

logger = logging.getLogger(__name__)

# Create router for meal endpoints
meal_router = APIRouter(prefix="/api/meals", tags=["Meal Search"])

# Pydantic models for meal endpoints
class MealSearchRequest(BaseModel):
    request: str = Field(..., min_length=1, max_length=500, description="User's meal request")
    max_results: int = Field(default=8, ge=1, le=20, description="Maximum number of meals to return")
    include_grocery: bool = Field(default=True, description="Include grocery shopping integration")
    cuisine_filter: Optional[List[str]] = Field(default=None, description="Filter by specific cuisines")
    diet_filter: Optional[List[str]] = Field(default=None, description="Filter by dietary requirements")
    max_cook_time: Optional[int] = Field(default=None, description="Maximum cooking time in minutes")
    budget_limit: Optional[float] = Field(default=None, description="Budget limit in EUR")

class MealPlanRequest(BaseModel):
    days: int = Field(..., ge=1, le=7, description="Number of days to plan")
    meals_per_day: int = Field(default=3, ge=1, le=5, description="Meals per day (breakfast, lunch, dinner, etc.)")
    people_count: int = Field(default=2, ge=1, le=10, description="Number of people")
    dietary_restrictions: Optional[List[str]] = Field(default=None, description="Dietary restrictions")
    budget_per_day: Optional[float] = Field(default=None, description="Budget per day in EUR")
    cuisine_variety: bool = Field(default=True, description="Include variety of cuisines")

class ShoppingListRequest(BaseModel):
    meal_ids: List[str] = Field(..., description="List of meal IDs to create shopping list for")
    people_count: int = Field(default=2, ge=1, le=10, description="Number of people")
    store_preference: Optional[str] = Field(default=None, description="Preferred store")

# Updated function definitions for meal search
MEAL_SEARCH_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_meals_by_request",
            "description": "Search for meals based on user's natural language request with grocery integration",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {"type": "string", "description": "User's meal request (e.g., 'healthy dinner for 4 people')"},
                    "max_results": {"type": "integer", "description": "Maximum number of meals to return", "default": 8},
                    "include_grocery": {"type": "boolean", "description": "Include grocery shopping integration", "default": True}
                },
                "required": ["user_request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_meal_plan",
            "description": "Create a comprehensive meal plan for multiple days with grocery integration",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of days to plan for"},
                    "meals_per_day": {"type": "integer", "description": "Number of meals per day", "default": 3},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 2},
                    "dietary_restrictions": {"type": "array", "items": {"type": "string"}, "description": "Dietary restrictions"},
                    "budget_per_day": {"type": "number", "description": "Budget per day in EUR"}
                },
                "required": ["days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_meal_recommendations_by_ingredients",
            "description": "Get meal recommendations based on available ingredients",
            "parameters": {
                "type": "object",
                "properties": {
                    "available_ingredients": {"type": "array", "items": {"type": "string"}, "description": "Ingredients user has available"},
                    "additional_shopping": {"type": "boolean", "description": "Allow additional ingredients from store", "default": True},
                    "max_additional_cost": {"type": "number", "description": "Maximum additional cost for extra ingredients"}
                },
                "required": ["available_ingredients"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_grocery_shopping_list_from_meals",
            "description": "Create optimized grocery shopping list from selected meals",
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_selections": {"type": "array", "items": {"type": "string"}, "description": "Selected meal titles or IDs"},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 2},
                    "store_preference": {"type": "string", "description": "Preferred store for shopping"}
                },
                "required": ["meal_selections"]
            }
        }
    }
]

# Meal search endpoints
@meal_router.post("/search")
async def search_meals(
    request: MealSearchRequest,
    grocery_mcp=Depends(lambda: None)  # You'll inject your grocery_mcp here
):
    """
    Search for meals based on user request with grocery integration
    """
    try:
        logger.info(f"🍽️ Meal search request: {request.request}")
        
        # Use the meal search system
        result = await search_meals_for_user(
            user_request=request.request,
            grocery_mcp=grocery_mcp,
            max_results=request.max_results
        )
        
        if result["success"]:
            # Apply additional filters if specified
            filtered_meals = await _apply_meal_filters(
                result["meals"], 
                request.cuisine_filter, 
                request.diet_filter,
                request.max_cook_time,
                request.budget_limit
            )
            
            return {
                "success": True,
                "data": {
                    "meals": filtered_meals,
                    "presentation": result["presentation"],
                    "search_info": {
                        "original_request": request.request,
                        "total_found": result["total_found"],
                        "after_filtering": len(filtered_meals),
                        "apis_used": result["apis_used"],
                        "grocery_integration": result["grocery_integration"]
                    }
                },
                "message": f"Found {len(filtered_meals)} meals for '{request.request}'"
            }
        else:
            return {
                "success": False,
                "data": {"meals": []},
                "message": "No meals found for your request",
                "suggestions": [
                    "Try broader search terms",
                    "Adjust dietary restrictions",
                    "Consider different cuisine types"
                ]
            }
    
    except Exception as e:
        logger.error(f"Meal search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Meal search failed: {str(e)}")

@meal_router.post("/plan")
async def create_meal_plan(
    request: MealPlanRequest,
    grocery_mcp=Depends(lambda: None)
):
    """
    Create a comprehensive meal plan for multiple days
    """
    try:
        logger.info(f"🗓️ Creating meal plan for {request.days} days, {request.people_count} people")
        
        meal_plan = await _generate_meal_plan(request, grocery_mcp)
        
        return {
            "success": True,
            "data": meal_plan,
            "message": f"Created {request.days}-day meal plan for {request.people_count} people"
        }
    
    except Exception as e:
        logger.error(f"Meal plan creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Meal plan creation failed: {str(e)}")

@meal_router.post("/shopping-list")
async def create_shopping_list(
    request: ShoppingListRequest,
    grocery_mcp=Depends(lambda: None)
):
    """
    Create optimized grocery shopping list from selected meals
    """
    try:
        logger.info(f"🛒 Creating shopping list for {len(request.meal_ids)} meals")
        
        shopping_list = await _create_optimized_shopping_list(request, grocery_mcp)
        
        return {
            "success": True,
            "data": shopping_list,
            "message": f"Created shopping list for {len(request.meal_ids)} meals"
        }
    
    except Exception as e:
        logger.error(f"Shopping list creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Shopping list creation failed: {str(e)}")

@meal_router.get("/recommendations/{cuisine}")
async def get_cuisine_recommendations(
    cuisine: str,
    limit: int = Query(default=6, ge=1, le=15),
    dietary_filter: Optional[str] = Query(default=None),
    grocery_mcp=Depends(lambda: None)
):
    """
    Get meal recommendations for a specific cuisine
    """
    try:
        # Build search request based on cuisine
        search_request = f"{cuisine} cuisine"
        if dietary_filter:
            search_request += f" {dietary_filter}"
        
        result = await search_meals_for_user(
            user_request=search_request,
            grocery_mcp=grocery_mcp,
            max_results=limit
        )
        
        return {
            "success": True,
            "data": {
                "cuisine": cuisine,
                "meals": result.get("meals", []),
                "recommendations": result.get("presentation", {}).get("recommendations", [])
            },
            "message": f"Found {len(result.get('meals', []))} {cuisine} meal recommendations"
        }
    
    except Exception as e:
        logger.error(f"Cuisine recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get {cuisine} recommendations: {str(e)}")

@meal_router.get("/dietary/{diet_type}")
async def get_dietary_meals(
    diet_type: str,
    limit: int = Query(default=8, ge=1, le=15),
    meal_type: Optional[str] = Query(default=None, description="breakfast, lunch, dinner, snack"),
    grocery_mcp=Depends(lambda: None)
):
    """
    Get meals for specific dietary requirements
    """
    try:
        # Build search request
        search_request = f"{diet_type} meals"
        if meal_type:
            search_request = f"{diet_type} {meal_type}"
        
        result = await search_meals_for_user(
            user_request=search_request,
            grocery_mcp=grocery_mcp,
            max_results=limit
        )
        
        return {
            "success": True,
            "data": {
                "diet_type": diet_type,
                "meal_type": meal_type,
                "meals": result.get("meals", []),
                "dietary_info": _extract_dietary_info(result.get("meals", []))
            },
            "message": f"Found {len(result.get('meals', []))} {diet_type} meals"
        }
    
    except Exception as e:
        logger.error(f"Dietary meals error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get {diet_type} meals: {str(e)}")

@meal_router.get("/quick/{max_minutes}")
async def get_quick_meals(
    max_minutes: int,
    limit: int = Query(default=8, ge=1, le=15),
    cuisine: Optional[str] = Query(default=None),
    grocery_mcp=Depends(lambda: None)
):
    """
    Get quick meals under specified time limit
    """
    try:
        search_request = f"quick meals under {max_minutes} minutes"
        if cuisine:
            search_request += f" {cuisine} cuisine"
        
        result = await search_meals_for_user(
            user_request=search_request,
            grocery_mcp=grocery_mcp,
            max_results=limit
        )
        
        # Filter by time
        quick_meals = []
        for meal in result.get("meals", []):
            total_time = getattr(meal, 'prep_time', 0) + getattr(meal, 'cook_time', 0)
            if total_time <= max_minutes:
                quick_meals.append(meal)
        
        return {
            "success": True,
            "data": {
                "max_minutes": max_minutes,
                "meals": quick_meals[:limit],
                "time_breakdown": _analyze_cooking_times(quick_meals)
            },
            "message": f"Found {len(quick_meals)} meals under {max_minutes} minutes"
        }
    
    except Exception as e:
        logger.error(f"Quick meals error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get quick meals: {str(e)}")

# Helper functions
async def _apply_meal_filters(
    meals: List,
    cuisine_filter: Optional[List[str]],
    diet_filter: Optional[List[str]],
    max_cook_time: Optional[int],
    budget_limit: Optional[float]
) -> List:
    """Apply additional filters to meal results"""
    
    filtered_meals = meals
    
    # Cuisine filter
    if cuisine_filter:
        filtered_meals = [
            meal for meal in filtered_meals
            if any(cuisine.lower() in getattr(meal, 'cuisine_type', '').lower() 
                   for cuisine in cuisine_filter)
        ]
    
    # Diet filter
    if diet_filter:
        filtered_meals = [
            meal for meal in filtered_meals
            if any(diet.lower() in [d.lower() for d in getattr(meal, 'diet_labels', [])]
                   for diet in diet_filter)
        ]
    
    # Time filter
    if max_cook_time:
        filtered_meals = [
            meal for meal in filtered_meals
            if (getattr(meal, 'prep_time', 0) + getattr(meal, 'cook_time', 0)) <= max_cook_time
        ]
    
    # Budget filter
    if budget_limit:
        filtered_meals = [
            meal for meal in filtered_meals
            if not getattr(meal, 'estimated_cost', None) or getattr(meal, 'estimated_cost', 0) <= budget_limit
        ]
    
    return filtered_meals

async def _generate_meal_plan(request: MealPlanRequest, grocery_mcp) -> Dict[str, Any]:
    """Generate a comprehensive meal plan"""
    
    meal_types = ["breakfast", "lunch", "dinner", "snack", "dessert"][:request.meals_per_day]
    plan = {"days": [], "total_cost": 0.0, "shopping_list": [], "nutrition_summary": {}}
    
    for day in range(1, request.days + 1):
        day_plan = {"day": day, "meals": [], "daily_cost": 0.0}
        
        for meal_type in meal_types:
            # Build search request for this meal
            search_request = f"{meal_type}"
            if request.dietary_restrictions:
                search_request += f" {' '.join(request.dietary_restrictions)}"
            search_request += f" for {request.people_count} people"
            
            # Search for meal
            result = await search_meals_for_user(
                user_request=search_request,
                grocery_mcp=grocery_mcp,
                max_results=3
            )
            
            if result["success"] and result["meals"]:
                selected_meal = result["meals"][0]  # Take best match
                
                # Calculate cost for this meal
                meal_cost = getattr(selected_meal, 'estimated_cost', 0) or 0
                if request.budget_per_day and meal_cost > (request.budget_per_day / request.meals_per_day):
                    # Try to find cheaper alternative
                    for alternative in result["meals"][1:]:
                        alt_cost = getattr(alternative, 'estimated_cost', 0) or 0
                        if alt_cost <= (request.budget_per_day / request.meals_per_day):
                            selected_meal = alternative
                            meal_cost = alt_cost
                            break
                
                day_plan["meals"].append({
                    "type": meal_type,
                    "meal": selected_meal,
                    "cost": meal_cost
                })
                day_plan["daily_cost"] += meal_cost
        
        plan["days"].append(day_plan)
        plan["total_cost"] += day_plan["daily_cost"]
    
    # Generate consolidated shopping list
    plan["shopping_list"] = await _consolidate_shopping_list(plan["days"], request.people_count)
    
    # Calculate nutrition summary
    plan["nutrition_summary"] = _calculate_nutrition_summary(plan["days"])
    
    return plan

async def _create_optimized_shopping_list(request: ShoppingListRequest, grocery_mcp) -> Dict[str, Any]:
    """Create optimized shopping list from selected meals"""
    
    # This would integrate with your grocery system to find best prices
    # For now, return a simplified version
    shopping_list = {
        "items": [],
        "total_cost": 0.0,
        "stores": {},
        "optimization_info": {
            "total_items": 0,
            "money_saved": 0.0,
            "best_store_distribution": {}
        }
    }
    
    # Here you would:
    # 1. Extract ingredients from all selected meals
    # 2. Consolidate duplicate ingredients
    # 3. Use grocery_mcp to find best prices for each ingredient
    # 4. Optimize store distribution for lowest total cost
    # 5. Calculate quantities for people_count
    
    return shopping_list

async def _consolidate_shopping_list(days: List[Dict], people_count: int) -> List[Dict]:
    """Consolidate ingredients from multiple days into optimized shopping list"""
    
    ingredient_map = {}
    
    for day in days:
        for meal_info in day["meals"]:
            meal = meal_info["meal"]
            shopping_list = getattr(meal, 'grocery_shopping_list', [])
            
            for item in shopping_list:
                ingredient_name = item.get("ingredient", "")
                if ingredient_name in ingredient_map:
                    # Combine quantities
                    ingredient_map[ingredient_name]["quantity"] += 1
                    ingredient_map[ingredient_name]["total_cost"] += item.get("estimated_cost", 0)
                else:
                    ingredient_map[ingredient_name] = {
                        "ingredient": ingredient_name,
                        "quantity": 1,
                        "total_cost": item.get("estimated_cost", 0),
                        "product_info": item.get("product", {})
                    }
    
    return list(ingredient_map.values())

def _calculate_nutrition_summary(days: List[Dict]) -> Dict[str, Any]:
    """Calculate nutrition summary for meal plan"""
    
    total_nutrition = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
    meal_count = 0
    
    for day in days:
        for meal_info in day["meals"]:
            meal = meal_info["meal"]
            nutrition = getattr(meal, 'nutrition', {})
            
            if nutrition:
                total_nutrition["calories"] += nutrition.get("calories", 0)
                total_nutrition["protein"] += float(str(nutrition.get("protein", 0)).replace("g", ""))
                total_nutrition["fat"] += float(str(nutrition.get("fat", 0)).replace("g", ""))
                total_nutrition["carbs"] += float(str(nutrition.get("carbs", 0)).replace("g", ""))
                meal_count += 1
    
    if meal_count > 0:
        return {
            "total": total_nutrition,
            "daily_average": {
                "calories": total_nutrition["calories"] / len(days),
                "protein": total_nutrition["protein"] / len(days),
                "fat": total_nutrition["fat"] / len(days),
                "carbs": total_nutrition["carbs"] / len(days)
            },
            "meal_count": meal_count
        }
    
    return {"total": total_nutrition, "daily_average": {}, "meal_count": 0}

def _extract_dietary_info(meals: List) -> Dict[str, Any]:
    """Extract dietary information from meals"""
    
    all_diets = set()
    all_allergens = set()
    
    for meal in meals:
        diet_labels = getattr(meal, 'diet_labels', [])
        allergen_info = getattr(meal, 'allergen_info', [])
        
        all_diets.update(diet_labels)
        all_allergens.update(allergen_info)
    
    return {
        "common_diets": list(all_diets),
        "allergens_present": list(all_allergens),
        "diet_distribution": _calculate_diet_distribution(meals)
    }

def _calculate_diet_distribution(meals: List) -> Dict[str, int]:
    """Calculate distribution of dietary labels across meals"""
    
    diet_count = {}
    
    for meal in meals:
        diet_labels = getattr(meal, 'diet_labels', [])
        for diet in diet_labels:
            diet_count[diet] = diet_count.get(diet, 0) + 1
    
    return diet_count

def _analyze_cooking_times(meals: List) -> Dict[str, Any]:
    """Analyze cooking time distribution"""
    
    times = []
    for meal in meals:
        total_time = getattr(meal, 'prep_time', 0) + getattr(meal, 'cook_time', 0)
        times.append(total_time)
    
    if not times:
        return {}
    
    return {
        "average_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "quick_meals": len([t for t in times if t <= 15]),
        "medium_meals": len([t for t in times if 15 < t <= 45]),
        "long_meals": len([t for t in times if t > 45])
    }

# Updated system message to include meal search capabilities
ENHANCED_SYSTEM_MESSAGE_WITH_MEALS = """You are an advanced AI grocery shopping assistant for Slovenia with revolutionary think-first approach AND comprehensive meal search capabilities.

🧠 **Your Think-First Approach:**
Instead of directly searching the database (which has missing data), you now use a much smarter approach PLUS you can search for meals using multiple recipe APIs and integrate them with grocery shopping.

**How Think-First + Meal Search Works:**
1. **Meal Search**: When users ask for meals, you search recipe APIs (Spoonacular, Edamam, TheMealDB)
2. **AI Product Generation**: Generate comprehensive grocery lists for found meals
3. **Targeted Database Search**: Search for each ingredient specifically in the Slovenian grocery database
4. **Complete Integration**: Provide meals with real Slovenian grocery prices and shopping lists

**Revolutionary Meal Functions:**
- 🍽️ **search_meals_by_request**: Search meals based on any natural language request
- 📅 **create_meal_plan**: Create multi-day meal plans with grocery integration
- 🛒 **create_grocery_shopping_list_from_meals**: Generate optimized shopping lists
- 🥗 **get_meal_recommendations_by_ingredients**: Suggest meals based on available ingredients

**Example of Meal + Grocery Integration:**
User: "Find healthy Italian dinner for 4 people"
1. Search recipe APIs for Italian dinner recipes
2. Filter for healthy options, 4 servings
3. Translate ingredients to Slovenian
4. Search Slovenian grocery database for each ingredient
5. Provide complete meal with ingredient prices from DM, Lidl, Mercator, SPAR, TUS

**Meal Search Capabilities:**
✅ **Multiple Recipe APIs** - Spoonacular, Edamam, TheMealDB
✅ **Natural Language Processing** - Understand complex meal requests
✅ **Dietary Filtering** - Vegetarian, vegan, keto, gluten-free, etc.
✅ **Cuisine Variety** - Italian, Asian, Mediterranean, Slovenian, etc.
✅ **Time-based Search** - Quick meals, elaborate dinners, meal prep
✅ **Grocery Integration** - Real Slovenian prices for all ingredients
✅ **Shopping List Generation** - Optimized across stores
✅ **Meal Planning** - Multi-day plans with budget optimization

**How to Help Users with Meals:**
1. **Always use meal search functions** for meal-related requests
2. **Explain the comprehensive approach** - from recipe APIs to grocery prices
3. **Provide complete solutions** - meal + grocery shopping + cost estimation
4. **Show variety and options** with different cuisines and dietary needs
5. **Include practical information** - cooking times, difficulty, nutrition

**Database Access for Groceries:**
- 34,790+ products from DM, Mercator, SPAR, TUS, LIDL
- AI-enhanced with health scores, nutrition grades, value ratings
- Think-first approach ensures comprehensive coverage
- Full integration with meal ingredient requirements

The think-first approach + meal search ensures you provide the most complete meal and grocery recommendations possible, combining international recipe knowledge with local Slovenian pricing data.

Respond in Slovenian when appropriate, and always highlight when the meal search + grocery integration provided comprehensive solutions.
"""
class MealDetailsRequest(BaseModel):
    meal_data: Dict[str, Any] = Field(..., description="Complete meal data from the meal card")


@meal_router.get("/details/{meal_id}")
async def get_meal_details(
    meal_id: str,
    meal_data: dict,  # Passed as JSON body
    grocery_mcp=Depends(lambda: None)
):
    """
    Get detailed meal information with grocery integration for a selected meal
    """
    try:
        logger.info(f"🛒 Getting detailed info for meal: {meal_id}")
        
        async with EnhancedMealSearchManager(grocery_mcp) as meal_manager:
            result = await meal_manager.get_meal_details_with_grocery(meal_id, meal_data)
        
        return {
            "success": result["success"],
            "data": result,
            "message": result.get("message", "Meal details retrieved")
        }
    
    except Exception as e:
        logger.error(f"Meal details error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get meal details: {str(e)}")


# Function to execute meal search functions
async def execute_meal_function(function_name: str, arguments: dict, grocery_mcp) -> dict:
    """Execute meal search functions"""
    try:
        if function_name == "search_meals_by_request":
            result = await search_meals_for_user(
                user_request=arguments["user_request"],
                grocery_mcp=grocery_mcp,
                max_results=arguments.get("max_results", 8)
            )
            return {"meal_search_result": result}
        
        elif function_name == "get_meal_details_with_grocery":
            # NEW: Handle meal details request
            async with EnhancedMealSearchManager(grocery_mcp) as meal_manager:
                result = await meal_manager.get_meal_details_with_grocery(
                    arguments["meal_id"], 
                    arguments["meal_data"]
                )
            return {"meal_details_result": result}
        
        elif function_name == "create_meal_plan":
            request = MealPlanRequest(**arguments)
            result = await _generate_meal_plan(request, grocery_mcp)
            return {"meal_plan_result": result}
        
        elif function_name == "get_meal_recommendations_by_ingredients":
            # Create search request based on available ingredients
            ingredients = arguments["available_ingredients"]
            search_request = f"meals using {', '.join(ingredients[:3])}"
            
            result = await search_meals_for_user(
                user_request=search_request,
                grocery_mcp=grocery_mcp,
                max_results=6
            )
            return {"ingredient_based_meals": result}
        
        elif function_name == "create_grocery_shopping_list_from_meals":
            request = ShoppingListRequest(
                meal_ids=arguments["meal_selections"],
                people_count=arguments.get("people_count", 2),
                store_preference=arguments.get("store_preference")
            )
            result = await _create_optimized_shopping_list(request, grocery_mcp)
            return {"shopping_list_result": result}
        
        else:
            raise ValueError(f"Unknown meal function: {function_name}")
    
    except Exception as e:
        logger.error(f"Error executing meal function {function_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Meal function execution failed: {str(e)}")

# Export the router and functions to integrate with your main FastAPI app
__all__ = [
    "meal_router", 
    "execute_meal_function", 
    "MEAL_SEARCH_FUNCTIONS", 
    "ENHANCED_SYSTEM_MESSAGE_WITH_MEALS"
]