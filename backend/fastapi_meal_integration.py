#!/usr/bin/env python3
"""
FastAPI Integration for Enhanced Meal Search
Adds meal search endpoints to your existing grocery intelligence backend
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Tuple
import asyncio
import logging

# Import your meal search system
from meal_search import EnhancedMealSearchManager, search_meals_for_user

logger = logging.getLogger(__name__)

# Create router for meal endpoints
meal_router = APIRouter(prefix="/api/meals", tags=["Meal Search"])

# Define APIResponse class
class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    approach: Optional[str] = None

# Pydantic models for meal endpoints
class MealSearchRequest(BaseModel):
    request: str = Field(..., min_length=1, max_length=500, description="User's meal request")
    max_results: int = Field(default=16, ge=1, le=20, description="Maximum number of meals to return")
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

class MealDetailsRequest(BaseModel):
    meal_data: Dict[str, Any] = Field(..., description="Complete meal data from the meal card")

class IngredientRequest(BaseModel):
    ingredients: List[Dict[str, Any]] = Field(..., description="List of ingredients to analyze")

# Dependency function (will be overridden in main.py)
async def get_enhanced_grocery_mcp():
    """Placeholder dependency - will be overridden in main.py"""
    return None

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
                    "max_results": {"type": "integer", "description": "Maximum number of meals to return", "default": 16},
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

# Cost analysis functions
def analyze_store_costs(ingredient_results: List[Dict]) -> Dict[str, Any]:
    """Analyze costs if shopping at individual stores"""
    stores = ['dm', 'lidl', 'mercator', 'spar', 'tus']
    store_analysis = {}
    
    for store in stores:
        total_cost = 0.0
        available_items = 0
        missing_items = []
        found_products = []
        
        for result in ingredient_results:
            ingredient = result['ingredient']
            search_results = result['search_result']
            
            # Find product in this specific store
            store_product = None
            for product in search_results:
                if (product.get('store_name', '').lower() == store.lower() and 
                    product.get('current_price') and 
                    product.get('current_price') > 0):
                    store_product = product
                    break
            
            if store_product:
                total_cost += store_product['current_price']
                available_items += 1
                found_products.append({
                    'ingredient': ingredient.get('name', ingredient.get('original', '')),
                    'product': store_product,
                    'price': store_product['current_price']
                })
            else:
                missing_items.append(ingredient.get('name', ingredient.get('original', '')))
        
        store_analysis[store] = {
            'store_name': store.upper(),
            'total_cost': round(total_cost, 2),
            'available_items': available_items,
            'missing_items': missing_items,
            'found_products': found_products,
            'completeness': round((available_items / len(ingredient_results)) * 100, 1) if ingredient_results else 0
        }
    
    return store_analysis

def analyze_combined_cheapest_costs(ingredient_results: List[Dict]) -> Dict[str, Any]:
    """Analyze costs using cheapest option for each ingredient across all stores"""
    total_cost = 0.0
    available_items = 0
    item_details = []
    
    for result in ingredient_results:
        ingredient = result['ingredient']
        search_results = result['search_result']
        
        if search_results:
            # Find the cheapest product across all stores
            valid_products = [
                p for p in search_results 
                if p.get('current_price') and p.get('current_price') > 0
            ]
            
            if valid_products:
                cheapest_product = min(valid_products, key=lambda x: x['current_price'])
                total_cost += cheapest_product['current_price']
                available_items += 1
                
                item_details.append({
                    'ingredient': ingredient.get('name', ingredient.get('original', '')),
                    'price': cheapest_product['current_price'],
                    'store': cheapest_product.get('store_name', ''),
                    'product': cheapest_product,
                    'found': True
                })
            else:
                item_details.append({
                    'ingredient': ingredient.get('name', ingredient.get('original', '')),
                    'price': None,
                    'store': None,
                    'product': None,
                    'found': False
                })
        else:
            item_details.append({
                'ingredient': ingredient.get('name', ingredient.get('original', '')),
                'price': None,
                'store': None,
                'product': None,
                'found': False
            })
    
    return {
        'total_cost': round(total_cost, 2),
        'available_items': available_items,
        'item_details': item_details,
        'completeness': round((available_items / len(ingredient_results)) * 100, 1) if ingredient_results else 0
    }

# Meal search endpoints
@meal_router.post("/search")
async def search_meals(
    request: MealSearchRequest,
    grocery_mcp=Depends(get_enhanced_grocery_mcp)
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

@meal_router.post("/details/{meal_id}")
async def get_meal_details(
    meal_id: str,
    request: MealDetailsRequest,
    grocery_mcp=Depends(get_enhanced_grocery_mcp)
):
    """
    Get detailed meal information with grocery integration for a selected meal
    """
    try:
        logger.info(f"🛒 Getting detailed info for meal: {meal_id}")
        
        async with EnhancedMealSearchManager(grocery_mcp) as meal_manager:
            result = await meal_manager.get_meal_details_with_grocery(
                meal_id, 
                request.meal_data
            )
        
        return {
            "success": result["success"],
            "data": result,
            "message": result.get("message", "Meal details retrieved")
        }
    
    except Exception as e:
        logger.error(f"Meal details error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get meal details: {str(e)}")

@meal_router.post("/cost-analysis")
async def analyze_meal_costs(
    request: IngredientRequest,
    grocery_mcp=Depends(get_enhanced_grocery_mcp)
):
    """
    Analyze grocery costs for meal ingredients across stores
    """
    try:
        logger.info(f"🛒 Analyzing costs for {len(request.ingredients)} ingredients")
        
        if not grocery_mcp:
            raise HTTPException(status_code=500, detail="Grocery system not available")
        
        # Search for each ingredient in the database
        ingredient_results = []
        
        for ingredient in request.ingredients:
            ingredient_name = ingredient.get('name', ingredient.get('original', ''))
            if not ingredient_name:
                continue
                
            try:
                # Use the enhanced search with validation
                search_result = await grocery_mcp.find_cheapest_product_with_intelligent_suggestions(
                    product_name=ingredient_name
                )
                
                ingredient_results.append({
                    'ingredient': ingredient,
                    'search_result': search_result.get('products', []) if search_result.get('success') else [],
                    'found': search_result.get('success', False) and len(search_result.get('products', [])) > 0
                })
                
            except Exception as e:
                logger.warning(f"Error searching for ingredient '{ingredient_name}': {e}")
                ingredient_results.append({
                    'ingredient': ingredient,
                    'search_result': [],
                    'found': False
                })
        
        # Analyze store-by-store costs
        store_analysis = analyze_store_costs(ingredient_results)
        
        # Analyze combined cheapest costs
        combined_analysis = analyze_combined_cheapest_costs(ingredient_results)
        
        return APIResponse(
            success=True,
            data={
                "store_analysis": store_analysis,
                "combined_analysis": combined_analysis,
                "ingredient_details": ingredient_results,
                "total_ingredients": len(request.ingredients),
                "found_ingredients": len([r for r in ingredient_results if r['found']])
            },
            message=f"Analyzed costs for {len(ingredient_results)} ingredients across Slovenian stores"
        )
        
    except Exception as e:
        logger.error(f"Cost analysis error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to analyze meal costs"
        )

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

# Function to execute meal search functions
async def execute_meal_function(function_name: str, arguments: dict, grocery_mcp) -> dict:
    """Execute meal search functions"""
    try:
        if function_name == "search_meals_by_request":
            result = await search_meals_for_user(
                user_request=arguments["user_request"],
                grocery_mcp=grocery_mcp,
                max_results=arguments.get("max_results", 16)
            )
            return {"meal_search_result": result}
        
        elif function_name == "get_meal_details_with_grocery":
            # Handle meal details request
            async with EnhancedMealSearchManager(grocery_mcp) as meal_manager:
                result = await meal_manager.get_meal_details_with_grocery(
                    arguments["meal_id"], 
                    arguments["meal_data"]
                )
            return {"meal_details_result": result}
        
        else:
            raise ValueError(f"Unknown meal function: {function_name}")
    
    except Exception as e:
        logger.error(f"Error executing meal function {function_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Meal function execution failed: {str(e)}")

# Enhanced system message
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

The think-first approach + meal search ensures you provide the most complete meal and grocery recommendations possible, combining international recipe knowledge with local Slovenian pricing data.

Respond in Slovenian when appropriate, and always highlight when the meal search + grocery integration provided comprehensive solutions.
"""

# Export the router and functions to integrate with your main FastAPI app
__all__ = [
    "meal_router", 
    "execute_meal_function", 
    "MEAL_SEARCH_FUNCTIONS", 
    "ENHANCED_SYSTEM_MESSAGE_WITH_MEALS",
    "analyze_store_costs",
    "analyze_combined_cheapest_costs",
    "APIResponse"
]