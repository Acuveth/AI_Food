#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Think-First Approach
Uses intelligent product generation before database search
"""

import os
import json
import logging
from typing import List, Optional, Any, Dict
from datetime import datetime
from contextlib import asynccontextmanager
import fastapi_meal_integration
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi_meal_integration import get_enhanced_grocery_mcp
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from fastapi_meal_integration import (
    meal_router, 
    execute_meal_function, 
    MEAL_SEARCH_FUNCTIONS, 
    ENHANCED_SYSTEM_MESSAGE_WITH_MEALS
)


# Import the enhanced systems
from grocery_intelligence import EnhancedSlovenianGroceryMCP
from database_source import EnhancedDatabaseSource, get_database_config
from semantic_search_validation import DynamicSemanticValidator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Database configuration
db_config = get_database_config()

# Global instances
enhanced_grocery_mcp = None
enhanced_db_source = None
dynamic_validator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global enhanced_grocery_mcp, enhanced_db_source, dynamic_validator
    
    # Startup
    logger.info("🚀 Starting Enhanced Slovenian Grocery Intelligence API with Think-First Approach...")
    
    # Initialize enhanced grocery MCP with think-first approach
    enhanced_grocery_mcp = EnhancedSlovenianGroceryMCP(db_config)
    await enhanced_grocery_mcp.connect_db()
    logger.info("✅ Enhanced Grocery MCP with think-first approach connected successfully")
    
    # Initialize enhanced database source
    enhanced_db_source = EnhancedDatabaseSource(db_config)
    await enhanced_db_source.connect()
    logger.info("✅ Enhanced database source connected successfully")
    
    # Initialize dynamic validator
    dynamic_validator = DynamicSemanticValidator()
    logger.info("✅ Dynamic LLM validator initialized")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Enhanced Slovenian Grocery Intelligence API...")
    if enhanced_grocery_mcp:
        enhanced_grocery_mcp.disconnect_db()
    if enhanced_db_source:
        enhanced_db_source.disconnect()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Slovenian Grocery Intelligence API",
    description="AI-powered grocery shopping assistant with think-first approach that generates intelligent product lists before searching",
    version="6.0.0",
    lifespan=lifespan
)

app.include_router(meal_router)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.dependency_overrides[fastapi_meal_integration.get_enhanced_grocery_mcp] = get_enhanced_grocery_mcp
# Enhanced OpenAI function definitions with think-first approach
ENHANCED_GROCERY_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "find_products_with_intelligent_validation",
            "description": "Find products using intelligent LLM-based validation that understands product intent",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to search for"},
                    "use_validation": {"type": "boolean", "description": "Whether to apply dynamic LLM validation", "default": True},
                    "store_preference": {"type": "string", "description": "Preferred store (dm, lidl, mercator, spar, tus)"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intelligent_health_focused_products",
            "description": "Get health-focused products using think-first approach: AI generates healthy product list, then searches database",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_health_score": {"type": "integer", "description": "Minimum health score (0-10)", "default": 7}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intelligent_diet_compatible_products",
            "description": "Find diet-compatible products using think-first approach: AI generates diet-specific product list, then searches database",
            "parameters": {
                "type": "object",
                "properties": {
                    "diet_type": {"type": "string", "description": "Diet type (vegan, vegetarian, keto, gluten_free, etc.)"}
                },
                "required": ["diet_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intelligent_meal_planning_suggestions",
            "description": "Get meal planning suggestions using think-first approach: AI generates meal-specific product list, then searches database",
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "description": "Type of meal"},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 1},
                    "budget": {"type": "number", "description": "Budget in EUR (optional)"}
                },
                "required": ["meal_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intelligent_smart_shopping_deals",
            "description": "Get smart shopping deals using think-first approach: AI generates deal-worthy product list, then searches database",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_deal_quality": {"type": "string", "enum": ["excellent", "good", "fair"], "default": "good"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intelligent_seasonal_recommendations",
            "description": "Get seasonal recommendations using think-first approach: AI generates seasonal product list, then searches database",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "string", "enum": ["spring", "summer", "autumn", "winter"], "description": "Season (auto-detected if not provided)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intelligent_allergen_safe_products",
            "description": "Get allergen-safe products using think-first approach: AI generates allergen-safe product list, then searches database",
            "parameters": {
                "type": "object",
                "properties": {
                    "avoid_allergens": {"type": "array", "items": {"type": "string"}, "description": "List of allergens to avoid (e.g., ['gluten', 'dairy', 'nuts'])"}
                },
                "required": ["avoid_allergens"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_intelligent_shopping_list",
            "description": "Create a shopping list using think-first approach: AI generates meal-specific product list, then searches database with budget optimization",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Budget in EUR"},
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "description": "Type of meal"},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 1}
                },
                "required": ["budget", "meal_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_prices_with_validation",
            "description": "Compare prices across stores with intelligent product validation",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to compare"},
                    "stores": {"type": "array", "items": {"type": "string"}, "description": "Specific stores to compare (optional)"},
                    "use_validation": {"type": "boolean", "description": "Apply dynamic validation", "default": True}
                },
                "required": ["product_name"]
            }
        }
    }
]
ENHANCED_GROCERY_FUNCTIONS.extend(MEAL_SEARCH_FUNCTIONS)


# Enhanced system message
ENHANCED_SYSTEM_MESSAGE = """You are an advanced AI grocery shopping assistant for Slovenia with revolutionary think-first approach.

🧠 **Your Revolutionary Think-First Approach:**
Instead of directly searching the database (which has missing data), you now use a much smarter approach:

**How Think-First Works:**
1. **AI Product Generation**: I first use my knowledge to generate a comprehensive list of products that should be available
2. **Targeted Database Search**: Then I search for each generated product specifically in the database
3. **Comparison with Database Functions**: Finally, I compare with traditional database functions
4. **Best Results**: I combine and present the best results from both approaches

**Why This Is Better:**
✅ **More Complete Results** - Database functions miss products due to incomplete data
✅ **Intelligent Product Selection** - AI knows what products should exist for each request
✅ **Better Coverage** - Searches for specific products rather than relying on database categories
✅ **Optimal Recommendations** - Combines AI knowledge with real database pricing

**Revolutionary Functions:**
- 🧠 **get_intelligent_health_focused_products**: AI generates healthy product list first
- 🥗 **get_intelligent_diet_compatible_products**: AI generates diet-specific products first  
- 🍽️ **get_intelligent_meal_planning_suggestions**: AI generates meal-specific products first
- 🛒 **get_intelligent_smart_shopping_deals**: AI generates deal-worthy products first
- 🌿 **get_intelligent_seasonal_recommendations**: AI generates seasonal products first
- 🛡️ **get_intelligent_allergen_safe_products**: AI generates allergen-safe products first

**Example of Think-First in Action:**
User: "Find healthy breakfast options"
1. AI generates: ["ovseni kosmiči", "grški jogurt", "borovnice", "banane", "mandljevo mleko"]
2. Database search: Finds actual products and prices for each
3. Comparison: Checks traditional health function for additional items
4. Result: Complete, comprehensive list with real prices

**Database Access:**
- 34,790+ products from DM, Mercator, SPAR, TUS, LIDL
- AI-enhanced with health scores, nutrition grades, value ratings
- Think-first approach ensures comprehensive coverage

**How to Help Users:**
1. **Always use intelligent functions** for specialized requests
2. **Explain the think-first approach** when relevant
3. **Highlight comprehensive results** from AI generation
4. **Show comparison insights** between generated and database results
5. **Provide complete recommendations** with pricing

**Example Response Pattern:**
"I used my think-first approach for healthy products: first generated a comprehensive list of 15 healthy items that should be available, then searched the database for each one. Found 12 of them with actual prices, plus 3 additional items from the database function. Here are the best healthy options with prices: [results]"

The think-first approach ensures you get the most complete and intelligent recommendations possible, combining AI knowledge with real-time pricing data.

Respond in Slovenian when appropriate, and always highlight when the think-first approach provided better results than traditional database functions would have.
"""

ENHANCED_SYSTEM_MESSAGE = ENHANCED_SYSTEM_MESSAGE_WITH_MEALS

ENHANCED_GROCERY_FUNCTIONS.append({
    "type": "function", 
    "function": {
        "name": "get_meal_details_with_grocery",
        "description": "Get detailed grocery information for a specific selected meal",
        "parameters": {
            "type": "object",
            "properties": {
                "meal_id": {"type": "string", "description": "ID of the selected meal"},
                "meal_data": {"type": "object", "description": "Basic meal data from the meal card"}
            },
            "required": ["meal_id", "meal_data"]
        }
    }
})


async def execute_enhanced_function(function_name: str, arguments: dict, mcp: EnhancedSlovenianGroceryMCP, db_source: EnhancedDatabaseSource) -> dict:
    """Execute enhanced grocery functions with think-first approach"""
    try:
         # Check if it's a meal function
        meal_functions = ["search_meals_by_request", "create_meal_plan", "get_meal_recommendations_by_ingredients", "create_grocery_shopping_list_from_meals", "get_meal_details_with_grocery" ]
        
        if function_name in meal_functions:
            return await execute_meal_function(function_name, arguments, mcp)

        if function_name in meal_functions:
            return await execute_meal_function(function_name, arguments, mcp)


        if function_name == "find_products_with_intelligent_validation":
            result = await mcp.find_cheapest_product_with_intelligent_suggestions(
                product_name=arguments["product_name"],
                store_preference=arguments.get("store_preference")
            )
            return result
        
        elif function_name == "get_intelligent_health_focused_products":
            result = await mcp.get_intelligent_health_focused_products(
                min_health_score=arguments.get("min_health_score", 7)
            )
            return {"intelligent_health_result": result}
        
        elif function_name == "get_intelligent_diet_compatible_products":
            result = await mcp.get_intelligent_diet_compatible_products(
                diet_type=arguments["diet_type"]
            )
            return {"intelligent_diet_result": result}
        
        elif function_name == "get_intelligent_meal_planning_suggestions":
            result = await mcp.get_intelligent_meal_planning_suggestions(
                meal_type=arguments["meal_type"],
                people_count=arguments.get("people_count", 1),
                budget=arguments.get("budget")
            )
            return {"intelligent_meal_result": result}
        
        elif function_name == "get_intelligent_smart_shopping_deals":
            result = await mcp.get_intelligent_smart_shopping_deals(
                min_deal_quality=arguments.get("min_deal_quality", "good")
            )
            return {"intelligent_deals_result": result}
        
        elif function_name == "get_intelligent_seasonal_recommendations":
            result = await mcp.get_intelligent_seasonal_recommendations(
                season=arguments.get("season")
            )
            return {"intelligent_seasonal_result": result}
        
        elif function_name == "get_intelligent_allergen_safe_products":
            result = await mcp.get_intelligent_allergen_safe_products(
                avoid_allergens=arguments["avoid_allergens"]
            )
            return {"intelligent_allergen_result": result}
        
        elif function_name == "create_intelligent_shopping_list":
            result = await mcp.get_intelligent_meal_planning_suggestions(
                meal_type=arguments["meal_type"],
                people_count=arguments.get("people_count", 1),
                budget=arguments["budget"]
            )
            return {"intelligent_shopping_list_result": result}
        
        elif function_name == "compare_prices_with_validation":
            result = await mcp.compare_prices(
                product_name=arguments["product_name"],
                stores=arguments.get("stores"),
                use_semantic_validation=arguments.get("use_validation", True)
            )
            return {"price_comparison": result}
        
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")

# Dependencies
async def get_enhanced_grocery_mcp() -> EnhancedSlovenianGroceryMCP:
    """Get enhanced grocery MCP instance"""
    if not enhanced_grocery_mcp:
        raise HTTPException(status_code=500, detail="Enhanced grocery system not initialized")
    return enhanced_grocery_mcp

async def get_enhanced_db_source() -> EnhancedDatabaseSource:
    """Get enhanced database source instance"""
    if not enhanced_db_source:
        raise HTTPException(status_code=500, detail="Enhanced database source not initialized")
    return enhanced_db_source

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    model: str = Field(default="gpt-4o-mini")

class ProductSearchRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=100)
    use_validation: bool = Field(default=True)
    store_preference: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    approach: Optional[str] = None


# New Pydantic models for meal cost analysis
class IngredientRequest(BaseModel):
    ingredients: List[Dict[str, Any]] = Field(..., description="List of ingredients to analyze")

class CostAnalysisResponse(BaseModel):
    success: bool
    store_analysis: Dict[str, Any]
    combined_analysis: Dict[str, Any]
    ingredient_details: List[Dict[str, Any]]
    message: Optional[str] = None


# Add this route to your FastAPI app in main.py
@app.post("/api/meals/cost-analysis", response_model=APIResponse)
async def analyze_meal_costs(
    request: IngredientRequest,
    mcp: EnhancedSlovenianGroceryMCP = Depends(get_enhanced_grocery_mcp)
):
    """
    Analyze grocery costs for meal ingredients across stores
    """
    try:
        logger.info(f"🛒 Analyzing costs for {len(request.ingredients)} ingredients")
        
        # Search for each ingredient in the database
        ingredient_results = []
        
        for ingredient in request.ingredients:
            ingredient_name = ingredient.get('name', ingredient.get('original', ''))
            if not ingredient_name:
                continue
                
            try:
                # Use the enhanced search with validation
                search_result = await mcp.find_cheapest_product_with_intelligent_suggestions(
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
        'completeness': round((available_items / len(ingredient_results)) * 100, 1) if ingredient_results else 0,
        'savings_vs_best_store': calculate_savings_vs_best_store(ingredient_results, total_cost)
    }

def calculate_savings_vs_best_store(ingredient_results: List[Dict], combined_cost: float) -> Dict[str, Any]:
    """Calculate how much the combined approach saves vs shopping at the best single store"""
    store_costs = analyze_store_costs(ingredient_results)
    
    # Find the best single store (lowest total cost for available items)
    best_store = None
    best_store_cost = float('inf')
    
    for store, analysis in store_costs.items():
        if analysis['available_items'] > 0 and analysis['total_cost'] < best_store_cost:
            best_store = store
            best_store_cost = analysis['total_cost']
    
    if best_store and best_store_cost < float('inf'):
        savings = best_store_cost - combined_cost
        return {
            'best_single_store': best_store.upper(),
            'best_store_cost': round(best_store_cost, 2),
            'combined_cost': round(combined_cost, 2),
            'savings_amount': round(savings, 2),
            'savings_percentage': round((savings / best_store_cost) * 100, 1) if best_store_cost > 0 else 0
        }
    
    return {
        'best_single_store': None,
        'best_store_cost': 0,
        'combined_cost': round(combined_cost, 2),
        'savings_amount': 0,
        'savings_percentage': 0
    }


# Enhanced Chat endpoint
@app.post("/api/chat", response_model=APIResponse)
async def enhanced_chat_with_gpt(
    message: ChatMessage,
    mcp: EnhancedSlovenianGroceryMCP = Depends(get_enhanced_grocery_mcp),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Enhanced chat with GPT using think-first approach"""
    try:
        # First GPT call with enhanced system message
        response = client.chat.completions.create(
            model=message.model,
            messages=[
                {"role": "system", "content": ENHANCED_SYSTEM_MESSAGE},
                {"role": "user", "content": message.message}
            ],
            tools=ENHANCED_GROCERY_FUNCTIONS,
            tool_choice="auto"
        )
        
        message_obj = response.choices[0].message
        
        # Check if GPT wants to call a function
        if message_obj.tool_calls:
            tool_call = message_obj.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the enhanced function
            function_result = await execute_enhanced_function(function_name, function_args, mcp, db_source)
            
            # Second GPT call with function result
            follow_up_response = client.chat.completions.create(
                model=message.model,
                messages=[
                    {"role": "system", "content": ENHANCED_SYSTEM_MESSAGE},
                    {"role": "user", "content": message.message},
                    {"role": "assistant", "content": message_obj.content, "tool_calls": message_obj.tool_calls},
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_result, ensure_ascii=False)}
                ],
                tools=ENHANCED_GROCERY_FUNCTIONS,
                tool_choice="auto"
            )
            
            final_message = follow_up_response.choices[0].message
            
            return APIResponse(
                success=True,
                data={
                    "response": final_message.content,
                    "function_used": function_name,
                    "function_result": function_result,
                    "think_first_approach": True
                },
                approach="think_first_then_search"
            )
        
        return APIResponse(
            success=True,
            data={
                "response": message_obj.content,
                "function_used": None,
                "function_result": None,
                "think_first_approach": True
            },
            approach="direct_response"
        )
    
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e),
            approach="error"
        )

# Enhanced search endpoint
@app.post("/api/search", response_model=APIResponse)
async def enhanced_product_search(
    request: ProductSearchRequest,
    mcp: EnhancedSlovenianGroceryMCP = Depends(get_enhanced_grocery_mcp)
):
    """Enhanced product search with dynamic LLM validation"""
    try:
        if request.use_validation:
            result = await mcp.find_cheapest_product_with_intelligent_suggestions(
                product_name=request.product_name,
                store_preference=request.store_preference
            )
            
            return APIResponse(
                success=result["success"],
                data={
                    "products": result.get("products", []),
                    "suggestions": result.get("suggestions", []),
                    "search_term": result.get("search_term"),
                    "validation_reasoning": result.get("validation_reasoning")
                },
                message=result.get("message"),
                approach="intelligent_search"
            )
        else:
            # Legacy search without validation
            products = await mcp.find_cheapest_product(
                product_name=request.product_name,
                store_preference=request.store_preference,
                use_semantic_validation=False
            )
            
            return APIResponse(
                success=len(products) > 0,
                data={"products": products},
                message=f"Found {len(products)} products (no validation applied)",
                approach="legacy_search"
            )
    
    except Exception as e:
        logger.error(f"Enhanced search error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e),
            approach="error"
        )

# Enhanced status endpoint
@app.get("/api/status", response_model=APIResponse)
async def get_system_status():
    """Get think-first system status"""
    try:
        global enhanced_grocery_mcp, enhanced_db_source, dynamic_validator
        
        status = {
            "think_first_approach_enabled": enhanced_grocery_mcp is not None,
            "approach_type": "Think-First Then Search",
            "revolutionary_features": [
                "AI generates comprehensive product lists first",
                "Targeted database search for each generated product", 
                "Comparison with traditional database functions",
                "Combined optimal results with real pricing",
                "Intelligent product knowledge beyond database categories",
                "Complete coverage despite missing database data"
            ],
            "intelligent_functions": [
                "get_intelligent_health_focused_products",
                "get_intelligent_diet_compatible_products", 
                "get_intelligent_meal_planning_suggestions",
                "get_intelligent_smart_shopping_deals",
                "get_intelligent_seasonal_recommendations",
                "get_intelligent_allergen_safe_products"
            ],
            "advantages_over_database_functions": [
                "More complete product coverage",
                "AI knowledge of what products should exist",
                "Not limited by database categorization",
                "Better results for specialized requests",
                "Combines AI intelligence with real pricing"
            ],
            "supported_languages": ["Slovenian", "English"],
            "llm_model": "gpt-4o-mini",
            "database_products": "34,790+",
            "stores": ["DM", "Lidl", "Mercator", "SPAR", "TUS"]
        }
        
        return APIResponse(success=True, data=status, approach="think_first_then_search")
    
    except Exception as e:
        return APIResponse(success=False, error=str(e), approach="error")

# Enhanced health check
@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "version": "6.0.0",
        "approach": "Think-First Then Search",
        "revolutionary_features": [
            "🧠 AI Product Generation First",
            "🔍 Targeted Database Search", 
            "📊 Database Function Comparison",
            "🎯 Optimal Combined Results",
            "💡 Intelligent Product Knowledge",
            "🛒 Complete Coverage Despite Missing Data",
            "🏥 AI Health Scoring",
            "💰 Smart Deal Analysis", 
            "🍽️ Diet Compatibility",
            "🌍 Environmental Impact",
            "📝 Adaptive Meal Planning"
        ],
        "database_connected": enhanced_db_source is not None,
        "grocery_mcp_connected": enhanced_grocery_mcp is not None,
        "think_first_enabled": enhanced_grocery_mcp is not None,
        "approach_improvements": [
            "🧠 AI generates comprehensive product lists first",
            "📊 Searches database for each generated product specifically",
            "🔄 Compares with traditional database function results",
            "🎯 Provides optimal combined recommendations",
            "💡 Overcomes database categorization limitations",
            "🚀 More complete results than database functions alone",
            "📈 Intelligent product knowledge beyond database scope"
        ],
        "example_workflow": [
            "User: 'Find healthy breakfast options'",
            "1. AI generates: ['ovseni kosmiči', 'grški jogurt', 'borovnice']",
            "2. Database search: Finds actual products and prices",
            "3. Comparison: Checks traditional health function",
            "4. Result: Complete list with best pricing"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Slovenian Grocery Intelligence API with Think-First Approach is running!", 
        "version": "6.0.0",
        "products_count": "34,790+",
        "approach": "Think-First Then Search",
        "revolutionary_upgrade": [
            "🧠 Think-First Approach - AI generates products before searching!",
            "📊 Database Content Analysis",
            "🎯 Targeted Product Search",
            "💡 Intelligent Product Generation",
            "📈 Optimal Combined Results",
            "🔄 Database Function Comparison"
        ],
        "why_think_first_is_better": [
            "Database functions miss products due to incomplete data",
            "AI knows what products should exist for each request",
            "Targeted search finds more relevant results",
            "Combines AI knowledge with real database pricing",
            "Not limited by database categorization"
        ],
        "example_improvements": [
            "Health requests: AI generates comprehensive healthy product list first",
            "Diet requests: AI generates diet-specific products then searches",
            "Meal planning: AI generates meal-specific shopping list then prices",
            "Seasonal recommendations: AI generates seasonal products then verifies availability"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )