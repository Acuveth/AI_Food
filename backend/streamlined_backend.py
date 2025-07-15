#!/usr/bin/env python3
"""
SILENT BACKEND VALIDATION - NO FRONTEND INDICATORS
AI validates and filters results internally without showing validation to users

This replaces your streamlined_backend.py with silent relevance validation:
- Evaluates every result for relevance
- Filters out bad matches automatically
- Sorts results by relevance (best first)
- Logs validation internally for monitoring
- Returns clean results without validation metadata
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import existing modules
from input_interpreter import interpret_user_input
from promotion_finder import find_promotions
from item_finder import compare_item_prices
from meal_search import search_meals, get_meal_with_grocery_analysis, reverse_meal_search
from database_handler import get_db_handler, close_db_handler

# Import the relevance evaluator for silent validation
from product_output_evaluator import ProductRelevanceEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global relevance evaluator for silent validation
relevance_evaluator = ProductRelevanceEvaluator()

# Validation settings
MIN_RELEVANCE_THRESHOLD = 40.0  # Filter out results below 40% relevance
ENABLE_SILENT_VALIDATION = True
ENABLE_RESULT_SORTING = True

# Pydantic models (unchanged)
class UserInputRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=500)

class PromotionRequest(BaseModel):
    search_filter: Optional[str] = None
    category_filter: Optional[str] = None
    store_filter: Optional[str] = None
    min_discount: Optional[int] = None
    max_price: Optional[float] = None
    sort_by: str = "discount_percentage"

class ItemComparisonRequest(BaseModel):
    item_name: str = Field(..., min_length=1)
    include_similar: bool = True
    max_results_per_store: int = 5

class MealSearchRequest(BaseModel):
    request: str = Field(..., min_length=1)
    max_results: int = 12

class MealGroceryRequest(BaseModel):
    meal_data: Dict[str, Any] = Field(...)

class ReverseMealRequest(BaseModel):
    ingredients: List[str] = Field(...)
    max_results: int = 10

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    intent: Optional[str] = None
    approach: Optional[str] = None

def silent_validate_and_filter(
    standard_result: Dict[str, Any], 
    user_query: str, 
    intent_type: str = "unknown"
) -> Dict[str, Any]:
    """
    SILENT VALIDATION: Evaluate relevance and filter results without frontend indicators
    """
    if not ENABLE_SILENT_VALIDATION:
        return standard_result
    
    try:
        # Evaluate relevance silently
        evaluation = relevance_evaluator.evaluate_system_output(user_query, standard_result, intent_type)
        
        # Log validation results for monitoring (internal only)
        logger.info(f"🔍 Silent validation: {evaluation.overall_relevance:.1f}/100 "
                   f"({evaluation.relevant_results}/{evaluation.total_results} relevant) "
                   f"Query: '{user_query[:50]}...'")
        
        # Filter and sort results based on relevance
        filtered_result = _filter_and_sort_results(standard_result, evaluation)
        
        # Log if we filtered anything
        original_count = evaluation.total_results
        final_count = _count_results(filtered_result)
        if final_count < original_count:
            logger.info(f"📊 Filtered {original_count - final_count} low-relevance results "
                       f"(kept {final_count}/{original_count})")
        
        # Warn about low overall quality (internal logging only)
        if evaluation.overall_relevance < 50:
            logger.warning(f"⚠️ Low relevance results for query: '{user_query}' "
                          f"(score: {evaluation.overall_relevance:.1f}/100)")
        
        return filtered_result
        
    except Exception as e:
        logger.error(f"❌ Silent validation failed: {e}")
        # Return original result if validation fails
        return standard_result

def _filter_and_sort_results(standard_result: Dict[str, Any], evaluation) -> Dict[str, Any]:
    """Filter out low-relevance results and sort by relevance"""
    if not evaluation.relevance_scores:
        return standard_result
    
    # Create relevance score lookup
    relevance_lookup = {}
    for i, score in enumerate(evaluation.relevance_scores):
        relevance_lookup[i] = score.overall_score
    
    # Filter and sort different result types
    filtered_result = standard_result.copy()
    
    # Handle promotions
    if "promotions" in filtered_result:
        promotions = filtered_result["promotions"]
        # Add relevance scores and filter
        filtered_promotions = []
        for i, promo in enumerate(promotions):
            relevance_score = relevance_lookup.get(i, 0)
            if relevance_score >= MIN_RELEVANCE_THRESHOLD:
                # Store relevance internally for sorting (not sent to frontend)
                promo["_internal_relevance"] = relevance_score
                filtered_promotions.append(promo)
        
        # Sort by relevance (best first) if enabled
        if ENABLE_RESULT_SORTING:
            filtered_promotions.sort(key=lambda x: x.get("_internal_relevance", 0), reverse=True)
        
        # Remove internal relevance scores before sending to frontend
        for promo in filtered_promotions:
            promo.pop("_internal_relevance", None)
        
        filtered_result["promotions"] = filtered_promotions
    
    # Handle meals
    if "meals" in filtered_result:
        meals = filtered_result["meals"]
        filtered_meals = []
        for i, meal in enumerate(meals):
            relevance_score = relevance_lookup.get(i, 0)
            if relevance_score >= MIN_RELEVANCE_THRESHOLD:
                meal["_internal_relevance"] = relevance_score
                filtered_meals.append(meal)
        
        if ENABLE_RESULT_SORTING:
            filtered_meals.sort(key=lambda x: x.get("_internal_relevance", 0), reverse=True)
        
        for meal in filtered_meals:
            meal.pop("_internal_relevance", None)
        
        filtered_result["meals"] = filtered_meals
    
    # Handle suggested_meals (reverse search)
    if "suggested_meals" in filtered_result:
        suggested_meals = filtered_result["suggested_meals"]
        filtered_suggested = []
        for i, meal in enumerate(suggested_meals):
            relevance_score = relevance_lookup.get(i, 0)
            if relevance_score >= MIN_RELEVANCE_THRESHOLD:
                meal["_internal_relevance"] = relevance_score
                filtered_suggested.append(meal)
        
        if ENABLE_RESULT_SORTING:
            filtered_suggested.sort(key=lambda x: x.get("_internal_relevance", 0), reverse=True)
        
        for meal in filtered_suggested:
            meal.pop("_internal_relevance", None)
        
        filtered_result["suggested_meals"] = filtered_suggested
    
    # Handle results_by_store (price comparison)
    if "results_by_store" in filtered_result:
        all_products = []
        for store_data in filtered_result["results_by_store"].values():
            all_products.extend(store_data.get("products", []))
        
        # Filter all products
        filtered_products = []
        for i, product in enumerate(all_products):
            relevance_score = relevance_lookup.get(i, 0)
            if relevance_score >= MIN_RELEVANCE_THRESHOLD:
                product["_internal_relevance"] = relevance_score
                filtered_products.append(product)
        
        if ENABLE_RESULT_SORTING:
            filtered_products.sort(key=lambda x: x.get("_internal_relevance", 0), reverse=True)
        
        # Redistribute filtered products back to stores
        store_products = {}
        for product in filtered_products:
            store_name = product.get("store_name", "unknown")
            if store_name not in store_products:
                store_products[store_name] = []
            # Remove internal relevance before adding
            product.pop("_internal_relevance", None)
            store_products[store_name].append(product)
        
        # Update store results
        for store_name, store_data in filtered_result["results_by_store"].items():
            store_data["products"] = store_products.get(store_name, [])
            store_data["product_count"] = len(store_data["products"])
    
    return filtered_result

def _count_results(result: Dict[str, Any]) -> int:
    """Count total results in a response"""
    count = 0
    if "promotions" in result:
        count += len(result["promotions"])
    elif "meals" in result:
        count += len(result["meals"])
    elif "suggested_meals" in result:
        count += len(result["suggested_meals"])
    elif "results_by_store" in result:
        for store_data in result["results_by_store"].values():
            count += len(store_data.get("products", []))
    return count

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting SILENTLY VALIDATED Grocery Intelligence API...")
    
    try:
        db_handler = await get_db_handler()
        logger.info("✅ Database connection established")
        logger.info("✅ Silent relevance validator initialized")
        logger.info(f"✅ Validation threshold: {MIN_RELEVANCE_THRESHOLD}% relevance")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise
    
    yield
    
    logger.info("🔄 Shutting down system...")
    await close_db_handler()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Silently Validated Grocery Intelligence API",
    description="AI-powered grocery shopping with silent relevance validation",
    version="3.0.0-silent",
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

# MAIN INTELLIGENT ENDPOINT - WITH SILENT VALIDATION
@app.post("/api/intelligent-request", response_model=APIResponse)
async def intelligent_request(request: UserInputRequest):
    """
    MAIN ENDPOINT: All workflows silently validated and filtered
    """
    try:
        logger.info(f"🧠 Processing silently validated request: '{request.input}'")
        
        # Step 1: Interpret user input
        interpretation = await interpret_user_input(request.input)
        intent = interpretation.get("intent")
        entities = interpretation.get("extracted_entities", {})
        
        logger.info(f"🎯 Intent: {intent}")
        
        # Step 2: Route to appropriate function and validate silently
        if intent == "FIND_PROMOTIONS":
            search_term = entities.get("search_term") or (entities.get("items", [None])[0])
            
            # Get standard results
            standard_result = await find_promotions(
                search_filter=search_term,
                category_filter=entities.get("category"),
                store_filter=entities.get("store_preference"),
                min_discount=entities.get("min_discount"),
                max_price=entities.get("max_price")
            )
            
            # SILENTLY VALIDATE AND FILTER
            validated_result = silent_validate_and_filter(
                standard_result, request.input, "FIND_PROMOTIONS"
            )
            
            return APIResponse(
                success=validated_result["success"],
                data=validated_result,
                message=validated_result.get("summary", "Promotions found"),
                intent=intent,
                approach="promotion_finder"
            )
        
        elif intent == "COMPARE_ITEM_PRICES":
            item_name = entities.get("search_term") or (entities.get("items", [""])[0])
            if not item_name:
                return APIResponse(
                    success=False,
                    error="No item specified for price comparison",
                    intent=intent
                )
            
            # Get standard results
            standard_result = await compare_item_prices(
                item_name=item_name,
                include_similar=True,
                max_results_per_store=5
            )
            
            # SILENTLY VALIDATE AND FILTER
            validated_result = silent_validate_and_filter(
                standard_result, request.input, "COMPARE_ITEM_PRICES"
            )
            
            return APIResponse(
                success=validated_result["success"],
                data=validated_result,
                message=validated_result.get("summary", "Price comparison completed"),
                intent=intent,
                approach="item_finder"
            )
        
        elif intent == "SEARCH_MEALS":
            # Get standard results
            standard_result = await search_meals(
                user_request=request.input,
                max_results=20
            )
            
            # SILENTLY VALIDATE AND FILTER
            validated_result = silent_validate_and_filter(
                standard_result, request.input, "SEARCH_MEALS"
            )
            
            return APIResponse(
                success=validated_result["success"],
                data=validated_result,
                message=validated_result.get("summary", "Meal search completed"),
                intent=intent,
                approach="meal_search"
            )
        
        elif intent == "REVERSE_MEAL_SEARCH":
            ingredients = entities.get("ingredients", [])
            if not ingredients:
                return APIResponse(
                    success=False,
                    error="No ingredients specified",
                    intent=intent
                )
            
            try:
                # Try the correct function from meal_search module
                from meal_search import reverse_meal_search
                standard_result = await reverse_meal_search(
                    available_ingredients=ingredients,
                    max_results=10
                )
            except ImportError:
                # Fallback if function doesn't exist
                logger.warning("reverse_meal_search function not found, using placeholder")
                standard_result = {
                    "success": True,
                    "suggested_meals": [],
                    "available_ingredients": ingredients,
                    "summary": "Reverse meal search temporarily unavailable"
                }
            
            # SILENTLY VALIDATE AND FILTER
            validated_result = silent_validate_and_filter(
                standard_result, request.input, "REVERSE_MEAL_SEARCH"
            )
            
            return APIResponse(
                success=validated_result["success"],
                data=validated_result,
                message=validated_result.get("summary", "Reverse meal search completed"),
                intent=intent,
                approach="reverse_meal_search"
            )
                
        else:
            # General response - no validation needed
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
        
    except Exception as e:
        logger.error(f"❌ Error in intelligent request: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to process request"
        )

# ALL DIRECT ENDPOINTS WITH SILENT VALIDATION

@app.post("/api/promotions", response_model=APIResponse)
async def get_promotions_validated(request: PromotionRequest):
    """Find promotional items with silent validation"""
    try:
        # Get standard results
        standard_result = await find_promotions(
            search_filter=request.search_filter,
            category_filter=request.category_filter,
            store_filter=request.store_filter,
            min_discount=request.min_discount,
            max_price=request.max_price,
            sort_by=request.sort_by
        )
        
        # SILENTLY VALIDATE AND FILTER
        # Use search_filter as user query for validation
        user_query = request.search_filter or "find promotions"
        validated_result = silent_validate_and_filter(
            standard_result, user_query, "FIND_PROMOTIONS"
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Promotions found"),
            approach="promotion_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Promotion search error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to find promotions"
        )

@app.post("/api/compare-prices", response_model=APIResponse)
async def compare_prices_validated(request: ItemComparisonRequest):
    """Compare item prices with silent validation"""
    try:
        # Get standard results
        standard_result = await compare_item_prices(
            item_name=request.item_name,
            include_similar=request.include_similar,
            max_results_per_store=request.max_results_per_store
        )
        
        # SILENTLY VALIDATE AND FILTER
        user_query = f"compare {request.item_name} prices"
        validated_result = silent_validate_and_filter(
            standard_result, user_query, "COMPARE_ITEM_PRICES"
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Price comparison completed"),
            approach="item_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Price comparison error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to compare prices"
        )

@app.post("/api/search-meals", response_model=APIResponse)
async def search_meals_validated(request: MealSearchRequest):
    """Search for meal recipes with silent validation"""
    try:
        # Get standard results
        standard_result = await search_meals(
            user_request=request.request,
            max_results=request.max_results
        )
        
        # SILENTLY VALIDATE AND FILTER
        validated_result = silent_validate_and_filter(
            standard_result, request.request, "SEARCH_MEALS"
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Meal search completed"),
            approach="meal_search"
        )
        
    except Exception as e:
        logger.error(f"❌ Meal search error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to search for meals"
        )

@app.post("/api/meal-grocery-analysis", response_model=APIResponse)
async def analyze_meal_grocery_validated(request: MealGroceryRequest):
    """Get grocery cost analysis (validation not applicable here)"""
    try:
        # Grocery analysis doesn't need relevance validation
        result = await get_meal_with_grocery_analysis(request.meal_data)
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("summary", "Grocery analysis completed"),
            approach="meal_grocery_analysis"
        )
        
    except Exception as e:
        logger.error(f"❌ Grocery analysis error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to analyze meal grocery costs"
        )

@app.post("/api/meals-from-ingredients", response_model=APIResponse)
async def find_meals_from_ingredients_validated(request: ReverseMealRequest):
    """Find meals from ingredients with silent validation"""
    try:
        # Get standard results
        standard_result = await reverse_meal_search(
            available_ingredients=request.ingredients,
            max_results=request.max_results
        )
        
        # SILENTLY VALIDATE AND FILTER
        user_query = f"meals with {', '.join(request.ingredients)}"
        validated_result = silent_validate_and_filter(
            standard_result, user_query, "REVERSE_MEAL_SEARCH"
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Reverse meal search completed"),
            approach="reverse_meal_search"
        )
        
    except Exception as e:
        logger.error(f"❌ Reverse meal search error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to find meals with your ingredients"
        )

# SIMPLE GET ENDPOINTS WITH SILENT VALIDATION
@app.get("/api/promotions/all", response_model=APIResponse)
async def get_all_promotions_validated(
    search: Optional[str] = Query(None, description="Search term"),
    store: Optional[str] = Query(None, description="Store filter"),
    min_discount: Optional[int] = Query(None, description="Minimum discount")
):
    """Get all promotions with silent validation"""
    try:
        # Get standard results
        standard_result = await find_promotions(
            search_filter=search,
            store_filter=store,
            min_discount=min_discount
        )
        
        # SILENTLY VALIDATE AND FILTER
        user_query = search or "find promotions"
        validated_result = silent_validate_and_filter(
            standard_result, user_query, "FIND_PROMOTIONS"
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "All promotions retrieved"),
            approach="promotion_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Get all promotions error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to retrieve promotions"
        )

@app.get("/api/compare-prices/{item_name}", response_model=APIResponse)
async def compare_prices_simple_validated(item_name: str):
    """Simple price comparison with silent validation"""
    try:
        # Get standard results
        standard_result = await compare_item_prices(
            item_name=item_name,
            include_similar=True,
            max_results_per_store=5
        )
        
        # SILENTLY VALIDATE AND FILTER
        user_query = f"compare {item_name} prices"
        validated_result = silent_validate_and_filter(
            standard_result, user_query, "COMPARE_ITEM_PRICES"
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Price comparison completed"),
            approach="item_finder"
        )
        
    except Exception as e:
        logger.error(f"❌ Simple price comparison error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Failed to compare prices"
        )

# UTILITY ENDPOINTS
@app.get("/api/health")
async def health_check():
    """Health check with silent validation status"""
    try:
        db_handler = await get_db_handler()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "version": "3.0.0-silent",
            "architecture": "silent_validation_system",
            "database_connected": db_handler is not None,
            "silent_validation": {
                "enabled": ENABLE_SILENT_VALIDATION,
                "threshold": f"{MIN_RELEVANCE_THRESHOLD}% relevance",
                "result_sorting": ENABLE_RESULT_SORTING,
                "coverage": "100% - ALL workflows silently validated"
            },
            "modules": [
                "input_interpreter",
                "promotion_finder", 
                "item_finder",
                "meal_search",
                "database_handler",
                "product_relevance_evaluator (silent)"
            ],
            "features": [
                "🔍 Silent relevance validation",
                "🚫 Automatic bad result filtering",
                "📊 Intelligent result sorting",
                "📈 Internal quality monitoring"
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
                "architecture": "silent_validation_system",
                "validation_mode": "SILENT - No frontend indicators",
                "core_functions": {
                    "promotion_finder": "Find promotional items (silently validated)",
                    "item_comparison": "Compare prices across stores (silently validated)",
                    "meal_search": "Search meals with analysis (silently validated)",
                    "grocery_analysis": "Cost analysis for meals",
                    "reverse_meal_search": "Find meals from ingredients (silently validated)"
                },
                "silent_validation_features": [
                    f"🎯 Relevance threshold: {MIN_RELEVANCE_THRESHOLD}%",
                    "🚫 Automatic bad result filtering",
                    "📊 Intelligent result sorting by relevance",
                    "📈 Internal quality monitoring",
                    "🔍 No frontend validation indicators"
                ],
                "database_status": "connected" if db_handler else "disconnected",
                "stores_supported": ["DM", "Lidl", "Mercator", "SPAR", "TUS"],
                "meal_apis": ["Spoonacular", "Edamam", "TheMealDB"]
            },
            message="System operational with SILENT relevance validation",
            approach="silent_validation_system"
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            error=str(e),
            message="System status check failed"
        )

@app.get("/")
async def root():
    """Root endpoint with silent validation info"""
    return {
        "message": "🛒 Silently Validated Grocery Intelligence API v3.0",
        "architecture": "Silent relevance validation - No frontend indicators",
        "validation_mode": "BACKEND ONLY - AI validates and filters internally",
        "core_functions": [
            {
                "name": "Intelligent Request Processing",
                "endpoint": "/api/intelligent-request",
                "description": "Natural language processing with silent validation"
            },
            {
                "name": "Promotion Finder", 
                "endpoint": "/api/promotions",
                "description": "Find deals with silent relevance filtering"
            },
            {
                "name": "Item Price Comparison",
                "endpoint": "/api/compare-prices", 
                "description": "Compare prices with silent validation"
            },
            {
                "name": "Meal Search & Analysis",
                "endpoints": ["/api/search-meals", "/api/meal-grocery-analysis", "/api/meals-from-ingredients"],
                "description": "Complete meal workflow with silent validation"
            }
        ],
        "silent_validation_features": [
            f"🎯 {MIN_RELEVANCE_THRESHOLD}% relevance threshold",
            "🚫 Automatic bad result filtering",
            "📊 Intelligent result sorting",
            "📈 Internal quality monitoring",
            "🔍 No frontend validation indicators"
        ],
        "user_experience": "Clean results without validation clutter",
        "getting_started": {
            "note": "All endpoints return clean, validated results",
            "examples": [
                "find cheap vegetarian milk",
                "compare organic bread prices", 
                "healthy Italian dinner for 4 people",
                "meals with chicken and vegetables"
            ]
        }
    }

# Error handlers (unchanged)
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return APIResponse(
        success=False,
        error="Endpoint not found",
        message="The requested endpoint does not exist"
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return APIResponse(
        success=False,
        error="Internal server error",
        message="An unexpected error occurred"
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