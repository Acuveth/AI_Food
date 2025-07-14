#!/usr/bin/env python3
"""
Streamlined FastAPI Backend
Main backend with 3 core functions: promotions, item comparison, and meal search
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our modules
from input_interpreter import interpret_user_input
from promotion_finder import find_promotions
from item_finder import compare_item_prices
from meal_search import search_meals, get_meal_with_grocery_analysis, reverse_meal_search
from database_handler import get_db_handler, close_db_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class UserInputRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=500, description="User's natural language input")

class PromotionRequest(BaseModel):
    search_filter: Optional[str] = Field(None, description="Filter promotional items by name")
    category_filter: Optional[str] = Field(None, description="Filter by category")
    store_filter: Optional[str] = Field(None, description="Filter by store")
    min_discount: Optional[int] = Field(None, description="Minimum discount percentage")
    max_price: Optional[float] = Field(None, description="Maximum price limit")
    sort_by: str = Field("discount_percentage", description="Sort criteria")

class ItemComparisonRequest(BaseModel):
    item_name: str = Field(..., min_length=1, description="Name of item to compare")
    include_similar: bool = Field(True, description="Include similar products")
    max_results_per_store: int = Field(5, description="Max results per store")

class MealSearchRequest(BaseModel):
    request: str = Field(..., min_length=1, description="Meal search request")
    max_results: int = Field(12, description="Maximum number of meals")

class MealGroceryRequest(BaseModel):
    meal_data: Dict[str, Any] = Field(..., description="Complete meal data")

class ReverseMealRequest(BaseModel):
    ingredients: List[str] = Field(..., description="Available ingredients")
    max_results: int = Field(10, description="Maximum meal suggestions")

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    intent: Optional[str] = None
    approach: Optional[str] = None

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting Streamlined Grocery Intelligence API...")
    
    # Initialize database connection
    try:
        db_handler = await get_db_handler()
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Streamlined Grocery Intelligence API...")
    await close_db_handler()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Streamlined Grocery Intelligence API",
    description="AI-powered grocery shopping with 3 core functions: promotions, price comparison, and meal search",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MAIN INTELLIGENT ENDPOINT
@app.post("/api/intelligent-request", response_model=APIResponse)
async def intelligent_request(request: UserInputRequest):
    """
    Main intelligent endpoint that interprets user input and routes to appropriate function
    """
    try:
        logger.info(f"🧠 Processing intelligent request: '{request.input}'")
        
        # Step 1: Interpret user input
        interpretation = await interpret_user_input(request.input)
        
        if not interpretation.get("intent"):
            return APIResponse(
                success=False,
                error="Failed to interpret user request",
                message="Could not understand what you're looking for",
                intent="unclear"
            )
        
        intent = interpretation["intent"]
        entities = interpretation.get("extracted_entities", {})
        
        logger.info(f"🎯 Detected intent: {intent}")
        
        # Step 2: Route to appropriate function based on intent
        if intent == "FIND_PROMOTIONS":
            # Safely extract search term
            search_term = entities.get("search_term")
            if not search_term:
                items_list = entities.get("items", [])
                search_term = items_list[0] if items_list and len(items_list) > 0 else None
            
            result = await find_promotions(
                search_filter=search_term,
                category_filter=entities.get("category"),
                store_filter=entities.get("store_preference"),
                min_discount=entities.get("min_discount"),
                max_price=entities.get("max_price")
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Promotions found"),
                intent=intent,
                approach="promotion_finder"
            )
        
        elif intent == "COMPARE_ITEM_PRICES":
            item_name = entities.get("search_term") or entities.get("items", [""])[0]
            if not item_name:
                return APIResponse(
                    success=False,
                    error="No item specified for price comparison",
                    message="Please specify which item you want to compare prices for",
                    intent=intent
                )
            
            result = await compare_item_prices(
                item_name=item_name,
                include_similar=True,
                max_results_per_store=5
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Price comparison completed"),
                intent=intent,
                approach="item_finder"
            )
        
        elif intent == "SEARCH_MEALS":
            result = await search_meals(
                user_request=request.input,
                max_results=20
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Meal search completed"),
                intent=intent,
                approach="meal_search"
            )
        
        elif intent == "REVERSE_MEAL_SEARCH":
            ingredients = entities.get("ingredients", [])
            if not ingredients:
                return APIResponse(
                    success=False,
                    error="No ingredients specified",
                    message="Please specify which ingredients you have available",
                    intent=intent
                )
            
            result = await reverse_meal_search(
                available_ingredients=ingredients,
                max_results=10
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Reverse meal search completed"),
                intent=intent,
                approach="reverse_meal_search"
            )
        
        elif intent == "GENERAL_QUESTION":
            return APIResponse(
                success=True,
                data={
                    "response": "I can help you find promotions, compare prices, or search for meals. Try asking something like 'find milk deals', 'compare bread prices', or 'Italian dinner recipes'.",
                    "suggestions": [
                        "Find promotional items",
                        "Compare prices across stores", 
                        "Search for meal recipes",
                        "Find meals with your ingredients"
                    ]
                },
                message="General assistance provided",
                intent=intent,
                approach="general_help"
            )
        
        else:  # UNCLEAR intent
            clarification_questions = await interpretation.get("clarification_questions", [
                "What specific product are you looking for?",
                "Would you like to find deals, compare prices, or get meal suggestions?",
                "Do you have any store preferences?"
            ])
            
            return APIResponse(
                success=False,
                data={
                    "clarification_questions": clarification_questions,
                    "suggestions": [
                        "Try: 'find milk promotions'",
                        "Try: 'compare cheese prices'",
                        "Try: 'healthy dinner recipes'",
                        "Try: 'meals with chicken and rice'"
                    ]
                },
                message="I need more information to help you",
                intent=intent,
                approach="clarification_needed"
            )
        
    except Exception as e:
        logger.error(f"❌ Error processing intelligent request: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to process your request",
            approach="error"
        )


@app.post("/api/intelligent-request", response_model=APIResponse)
async def intelligent_request(request: UserInputRequest):
    """
    Main intelligent endpoint that interprets user input and routes to appropriate function
    """
    try:
        logger.info(f"🧠 Processing intelligent request: '{request.input}'")
        
        # Step 1: Interpret user input
        interpretation = await interpret_user_input(request.input)
        
        if not interpretation.get("intent"):
            return APIResponse(
                success=False,
                error="Failed to interpret user request",
                message="Could not understand what you're looking for",
                intent="unclear"
            )
        
        intent = interpretation["intent"]
        entities = interpretation.get("extracted_entities", {})
        
        logger.info(f"🎯 Detected intent: {intent}")
        
        # Helper function to safely get search term
        def get_search_term():
            search_term = entities.get("search_term")
            if search_term:
                return search_term
            
            items_list = entities.get("items", [])
            if items_list and len(items_list) > 0:
                return items_list[0]
            
            return None
        
        # Step 2: Route to appropriate function based on intent
        if intent == "FIND_PROMOTIONS":
            search_term = get_search_term()
            
            result = await find_promotions(
                search_filter=search_term,
                category_filter=entities.get("category"),
                store_filter=entities.get("store_preference"),
                min_discount=entities.get("min_discount"),
                max_price=entities.get("max_price")
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Promotions found"),
                intent=intent,
                approach="promotion_finder"
            )
        
        elif intent == "COMPARE_ITEM_PRICES":
            item_name = get_search_term()
            if not item_name:
                return APIResponse(
                    success=False,
                    error="No item specified for price comparison",
                    message="Please specify which item you want to compare prices for",
                    intent=intent
                )
            
            result = await compare_item_prices(
                item_name=item_name,
                include_similar=True,
                max_results_per_store=5
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Price comparison completed"),
                intent=intent,
                approach="item_finder"
            )
        
        elif intent == "SEARCH_MEALS":
            result = await search_meals(
                user_request=request.input,
                max_results=20
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Meal search completed"),
                intent=intent,
                approach="meal_search"
            )
        
        elif intent == "REVERSE_MEAL_SEARCH":
            ingredients = entities.get("ingredients", [])
            if not ingredients:
                return APIResponse(
                    success=False,
                    error="No ingredients specified",
                    message="Please specify which ingredients you have available",
                    intent=intent
                )
            
            result = await reverse_meal_search(
                available_ingredients=ingredients,
                max_results=10
            )
            
            return APIResponse(
                success=result["success"],
                data=result,
                message=result.get("summary", "Reverse meal search completed"),
                intent=intent,
                approach="reverse_meal_search"
            )
        
        elif intent == "GENERAL_QUESTION":
            return APIResponse(
                success=True,
                data={
                    "response": "I can help you find promotions, compare prices, or search for meals. Try asking something like 'find milk deals', 'compare bread prices', or 'Italian dinner recipes'.",
                    "suggestions": [
                        "Find promotional items",
                        "Compare prices across stores", 
                        "Search for meal recipes",
                        "Find meals with your ingredients"
                    ]
                },
                message="General assistance provided",
                intent=intent,
                approach="general_help"
            )
        
        else:  # UNCLEAR intent
            clarification_questions = interpretation.get("clarification_questions", [
                "What specific product are you looking for?",
                "Would you like to find deals, compare prices, or get meal suggestions?",
                "Do you have any store preferences?"
            ])
            
            return APIResponse(
                success=False,
                data={
                    "clarification_questions": clarification_questions,
                    "suggestions": [
                        "Try: 'find milk promotions'",
                        "Try: 'compare cheese prices'",
                        "Try: 'healthy dinner recipes'",
                        "Try: 'meals with chicken and rice'"
                    ]
                },
                message="I need more information to help you",
                intent=intent,
                approach="clarification_needed"
            )
        
    except Exception as e:
        logger.error(f"❌ Error processing intelligent request: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to process your request",
            approach="error"
        )


# CORE FUNCTION 1: PROMOTION FINDER
@app.post("/api/promotions", response_model=APIResponse)
async def get_promotions(request: PromotionRequest):
    """
    Core Function 1: Find promotional items with optional filtering
    """
    try:
        logger.info(f"🏷️ Finding promotions with filters: {request.search_filter}")
        
        result = await find_promotions(
            search_filter=request.search_filter,
            category_filter=request.category_filter,
            store_filter=request.store_filter,
            min_discount=request.min_discount,
            max_price=request.max_price,
            sort_by=request.sort_by
        )
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Promotions search completed"),
            approach="promotion_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Promotion search error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to find promotions",
            approach="error"
        )

# Simple GET endpoint for promotions without filters
@app.get("/api/promotions/all", response_model=APIResponse)
async def get_all_promotions(
    search: Optional[str] = Query(None, description="Search term to filter promotions"),
    store: Optional[str] = Query(None, description="Store filter"),
    min_discount: Optional[int] = Query(None, description="Minimum discount percentage")
):
    """Get all promotions with simple query parameters"""
    try:
        result = await find_promotions(
            search_filter=search,
            store_filter=store,
            min_discount=min_discount
        )
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "All promotions retrieved"),
            approach="promotion_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Get all promotions error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to retrieve promotions",
            approach="error"
        )

# CORE FUNCTION 2: ITEM PRICE COMPARISON
@app.post("/api/compare-prices", response_model=APIResponse)
async def compare_prices(request: ItemComparisonRequest):
    """
    Core Function 2: Compare item prices across all stores
    """
    try:
        logger.info(f"🔍 Comparing prices for: {request.item_name}")
        
        result = await compare_item_prices(
            item_name=request.item_name,
            include_similar=request.include_similar,
            max_results_per_store=request.max_results_per_store
        )
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Price comparison completed"),
            approach="item_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Price comparison error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to compare prices",
            approach="error"
        )

# Simple GET endpoint for price comparison
@app.get("/api/compare-prices/{item_name}", response_model=APIResponse)
async def compare_prices_simple(item_name: str):
    """Simple GET endpoint for price comparison"""
    try:
        result = await compare_item_prices(
            item_name=item_name,
            include_similar=True,
            max_results_per_store=5
        )
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Price comparison completed"),
            approach="item_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Simple price comparison error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to compare prices",
            approach="error"
        )

# CORE FUNCTION 3: MEAL SEARCH AND GROCERY INTEGRATION
@app.post("/api/search-meals", response_model=APIResponse)
async def search_meal_recipes(request: MealSearchRequest):
    """
    Core Function 3a: Search for meal recipes based on user request
    """
    try:
        logger.info(f"🍽️ Searching meals for: {request.request}")
        
        result = await search_meals(
            user_request=request.request,
            max_results=request.max_results
        )
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Meal search completed"),
            approach="meal_search"
        )
        
    except Exception as e:
        logger.error(f"❌ Meal search error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to search for meals",
            approach="error"
        )

@app.post("/api/meal-grocery-analysis", response_model=APIResponse)
async def analyze_meal_grocery_costs(request: MealGroceryRequest):
    """
    Core Function 3b: Get grocery cost analysis for selected meal
    """
    try:
        meal_title = request.meal_data.get("title", "selected meal")
        logger.info(f"🛒 Analyzing grocery costs for: {meal_title}")
        
        result = await get_meal_with_grocery_analysis(request.meal_data)
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Grocery analysis completed"),
            approach="meal_grocery_analysis"
        )
        
    except Exception as e:
        logger.error(f"❌ Meal grocery analysis error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to analyze meal grocery costs",
            approach="error"
        )

@app.post("/api/meals-from-ingredients", response_model=APIResponse)
async def find_meals_from_ingredients(request: ReverseMealRequest):
    """
    Core Function 3c: Find meals that can be made with available ingredients
    """
    try:
        logger.info(f"🔍 Finding meals with ingredients: {request.ingredients}")
        
        result = await reverse_meal_search(
            available_ingredients=request.ingredients,
            max_results=request.max_results
        )
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Reverse meal search completed"),
            approach="reverse_meal_search"
        )
        
    except Exception as e:
        logger.error(f"❌ Reverse meal search error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to find meals with your ingredients",
            approach="error"
        )

# UTILITY ENDPOINTS
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_handler = await get_db_handler()
        db_connected = db_handler is not None
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "version": "2.0.0",
            "architecture": "streamlined_modular",
            "database_connected": db_connected,
            "modules": [
                "input_interpreter",
                "promotion_finder", 
                "item_finder",
                "meal_search",
                "database_handler"
            ],
            "core_functions": [
                "🏷️ Promotion Finder - Find discounted items",
                "🔍 Item Price Comparison - Compare prices across stores",
                "🍽️ Meal Search - Find recipes with grocery integration"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }

@app.get("/api/status", response_model=APIResponse)
async def get_system_status():
    """Get detailed system status"""
    try:
        db_handler = await get_db_handler()
        
        return APIResponse(
            success=True,
            data={
                "system_status": "operational",
                "architecture": "streamlined_modular",
                "core_functions": {
                    "promotion_finder": "Find promotional items with intelligent filtering",
                    "item_comparison": "Compare item prices across all Slovenian stores",
                    "meal_search": "Search meals with grocery cost analysis"
                },
                "intelligent_features": [
                    "🧠 Natural language input interpretation",
                    "🎯 Automatic intent detection and routing",
                    "💡 LLM-powered product search suggestions",
                    "📊 Advanced price and promotion analysis",
                    "🛒 Complete meal-to-grocery integration"
                ],
                "database_status": "connected" if db_handler else "disconnected",
                "stores_supported": ["DM", "Lidl", "Mercator", "SPAR", "TUS"],
                "meal_apis": ["Spoonacular", "Edamam", "TheMealDB"]
            },
            message="System operational with all modules ready",
            approach="streamlined_modular"
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            error=str(e),
            message="System status check failed",
            approach="error"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🛒 Streamlined Grocery Intelligence API v2.0",
        "architecture": "Modular with intelligent input interpretation",
        "core_functions": [
            {
                "name": "Intelligent Request Processing",
                "endpoint": "/api/intelligent-request",
                "description": "Main endpoint that interprets natural language and routes to appropriate function"
            },
            {
                "name": "Promotion Finder", 
                "endpoint": "/api/promotions",
                "description": "Find discounted items across all stores with advanced filtering"
            },
            {
                "name": "Item Price Comparison",
                "endpoint": "/api/compare-prices", 
                "description": "Compare prices for specific items across all stores"
            },
            {
                "name": "Meal Search & Grocery Integration",
                "endpoints": ["/api/search-meals", "/api/meal-grocery-analysis", "/api/meals-from-ingredients"],
                "description": "Search meals, analyze grocery costs, and reverse meal search"
            }
        ],
        "features": [
            "🧠 Natural language understanding",
            "🏷️ Smart promotion finding",
            "🔍 Comprehensive price comparison", 
            "🍽️ Meal search with grocery integration",
            "📊 LLM-powered analysis and insights",
            "🛒 Complete shopping cost breakdown"
        ],
        "getting_started": {
            "try_intelligent_request": "POST /api/intelligent-request with any natural language input",
            "examples": [
                "find milk deals",
                "compare bread prices", 
                "healthy Italian dinner recipes",
                "meals with chicken and rice"
            ]
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return APIResponse(
        success=False,
        error="Endpoint not found",
        message="The requested endpoint does not exist",
        approach="error"
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return APIResponse(
        success=False,
        error="Internal server error",
        message="An unexpected error occurred",
        approach="error"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "streamlined_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )