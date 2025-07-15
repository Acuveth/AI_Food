#!/usr/bin/env python3
"""
SLOVENIAN BACKEND VALIDATION with Slovenian Response Messages
Enhanced with comprehensive Slovenian language support for all user interactions
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

# Slovenian response messages
SLOVENIAN_MESSAGES = {
    "welcome": "Dobrodošli v sistem za pametno nakupovanje!",
    "no_promotions": "Ni najdenih akcij za vaše iskanje.",
    "no_items": "Ni najdenih izdelkov za primerjavo cen.",
    "no_meals": "Ni najdenih jedi za vaš zahtevek.",
    "no_ingredients": "Ni specificirane sestavine.",
    "processing": "Obdelujem vašo zahtevo...",
    "error": "Napaka pri obdelavi zahteve.",
    "connection_error": "Napaka pri povezavi z bazo podatkov.",
    "promotions_found": "Najdene akcije",
    "price_comparison_completed": "Primerjava cen dokončana",
    "meal_search_completed": "Iskanje jedi dokončano",
    "grocery_analysis_completed": "Analiza stroškov nakupovanja dokončana",
    "reverse_meal_search_completed": "Iskanje jedi z vašimi sestavinami dokončano",
    "general_help": "Splošna pomoč",
    "clarification_needed": "Potrebno je pojasnilo",
    "search_suggestions": [
        "Poskusite z drugačnimi iskalnimi pojmi",
        "Uporabite slovenska imena izdelkov",
        "Omenite lahko trgovino (DM, Lidl, Mercator, SPAR, Tuš)",
        "Sprostite prehranske omejitve"
    ],
    "general_response": "Pomagam vam najti akcije, primerjati cene ali iskati recepte. Poskusite vprašati nekaj kot 'najdi akcije za mleko', 'primerjaj cene kruha' ali 'italijanski recepti za večerjo'.",
    "general_suggestions": [
        "Najdi akcije za mleko",
        "Primerjaj cene kruha v trgovinah",
        "Vegetarijski recepti za kosilo",
        "Kaj lahko skuham s piščancem in rižem"
    ]
}

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

def get_slovenian_message(key: str, default: str = None) -> str:
    """Get Slovenian message by key with fallback"""
    return SLOVENIAN_MESSAGES.get(key, default or key)

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
        logger.info(f"🔍 Tiha validacija: {evaluation.overall_relevance:.1f}/100 "
                   f"({evaluation.relevant_results}/{evaluation.total_results} relevantnih) "
                   f"Poizvedba: '{user_query[:50]}...'")
        
        # Filter and sort results based on relevance
        filtered_result = _filter_and_sort_results(standard_result, evaluation)
        
        # Log if we filtered anything
        original_count = evaluation.total_results
        final_count = _count_results(filtered_result)
        if final_count < original_count:
            logger.info(f"📊 Filtrirani {original_count - final_count} nizko relevantni rezultati "
                       f"(ohranjen {final_count}/{original_count})")
        
        # Warn about low overall quality (internal logging only)
        if evaluation.overall_relevance < 50:
            logger.warning(f"⚠️ Nizko relevantni rezultati za poizvedbo: '{user_query}' "
                          f"(ocena: {evaluation.overall_relevance:.1f}/100)")
        
        return filtered_result
        
    except Exception as e:
        logger.error(f"❌ Tiha validacija ni uspela: {e}")
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
    logger.info("🚀 Zagon sistema za pametno nakupovanje z slovensko podporo...")
    
    try:
        db_handler = await get_db_handler()
        if db_handler:
            logger.info("✅ Povezava z bazo podatkov vzpostavljena")
            logger.info("✅ Tihi validator relevantnosti inicializiran")
            logger.info(f"✅ Próg validacije: {MIN_RELEVANCE_THRESHOLD}% relevantnost")
    except Exception as e:
        logger.error(f"❌ Neuspešna inicializacija sistema: {e}")
        raise
    
    yield
    
    logger.info("🔄 Zaustavitev sistema...")
    await close_db_handler()
    logger.info("✅ Zaustavitev dokončana")

# Create FastAPI app
app = FastAPI(
    title="Sistem za pametno nakupovanje - Slovenija",
    description="AI-powered grocery shopping with Slovenian language support",
    version="3.0.0-slovenian",
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

# MAIN INTELLIGENT ENDPOINT - WITH SLOVENIAN SUPPORT
@app.post("/api/intelligent-request", response_model=APIResponse)
async def intelligent_request(request: UserInputRequest):
    """
    GLAVNI ENDPOINT: Vsi delovni procesi z validacijo in slovensko podporo
    """
    try:
        logger.info(f"🧠 Obdelava zahteve z slovensko podporo: '{request.input}'")
        
        # Step 1: Interpret user input with Slovenian support
        interpretation = await interpret_user_input(request.input)
        intent = interpretation.get("intent")
        entities = interpretation.get("extracted_entities", {})
        
        logger.info(f"🎯 Namen: {intent}")
        
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
                message=validated_result.get("summary", get_slovenian_message("promotions_found")),
                intent=intent,
                approach="promotion_finder"
            )
        
        elif intent == "COMPARE_ITEM_PRICES":
            item_name = entities.get("search_term") or (entities.get("items", [""])[0])
            if not item_name:
                return APIResponse(
                    success=False,
                    error=get_slovenian_message("no_items"),
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
                message=validated_result.get("summary", get_slovenian_message("price_comparison_completed")),
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
                message=validated_result.get("summary", get_slovenian_message("meal_search_completed")),
                intent=intent,
                approach="meal_search"
            )
        
        elif intent == "REVERSE_MEAL_SEARCH":
            ingredients = entities.get("ingredients", [])
            if not ingredients:
                return APIResponse(
                    success=False,
                    error=get_slovenian_message("no_ingredients"),
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
                logger.warning("reverse_meal_search funkcija ni najdena, uporaba nadomestka")
                standard_result = {
                    "success": True,
                    "suggested_meals": [],
                    "available_ingredients": ingredients,
                    "summary": "Iskanje jedi z vašimi sestavinami trenutno ni na voljo"
                }
            
            # SILENTLY VALIDATE AND FILTER
            validated_result = silent_validate_and_filter(
                standard_result, request.input, "REVERSE_MEAL_SEARCH"
            )
            
            return APIResponse(
                success=validated_result["success"],
                data=validated_result,
                message=validated_result.get("summary", get_slovenian_message("reverse_meal_search_completed")),
                intent=intent,
                approach="reverse_meal_search"
            )
                
        else:
            # General response - no validation needed
            return APIResponse(
                success=True,
                data={
                    "response": get_slovenian_message("general_response"),
                    "suggestions": get_slovenian_message("general_suggestions")
                },
                message=get_slovenian_message("general_help"),
                intent=intent,
                approach="general_help"
            )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri obdelavi zahteve: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message=get_slovenian_message("error")
        )

@app.get("/api/health")
async def health_check():
    """Health check s slovensko podporo"""
    try:
        db_handler = await get_db_handler()
        
        return {
            "status": "zdrav",
            "timestamp": datetime.now(),
            "version": "3.0.0-slovenian",
            "architecture": "sistem_tihe_validacije",
            "database_connected": db_handler is not None,
            "silent_validation": {
                "enabled": ENABLE_SILENT_VALIDATION,
                "threshold": f"{MIN_RELEVANCE_THRESHOLD}% relevantnost",
                "result_sorting": ENABLE_RESULT_SORTING,
                "coverage": "100% - VSI delovni procesi tiho validirani"
            },
            "modules": [
                "input_interpreter (slovenska podpora)",
                "promotion_finder (slovenska podpora)", 
                "item_finder (slovenska podpora)",
                "meal_search (slovenska podpora)",
                "database_handler (slovenska podpora)",
                "product_relevance_evaluator (tiha validacija)"
            ],
            "supported_languages": ["slovenščina", "angleščina"],
            "supported_stores": ["DM", "Lidl", "Mercator", "SPAR", "Tuš"],
            "features": [
                "🔍 Tiha validacija relevantnosti",
                "🚫 Avtomatsko filtriranje slabih rezultatov",
                "📊 Inteligentno razvrščanje rezultatov",
                "📈 Notranje spremljanje kakovosti",
                "🇸🇮 Popolna slovenska podpora"
            ]
        }
        
    except Exception as e:
        return {
            "status": "nezdrav",
            "error": str(e),
            "timestamp": datetime.now()
        }

@app.get("/api/status", response_model=APIResponse)
async def get_system_status():
    """Podroben status sistema z slovensko podporo"""
    try:
        db_handler = await get_db_handler()
        
        return APIResponse(
            success=True,
            data={
                "system_status": "operativen",
                "architecture": "sistem_tihe_validacije",
                "validation_mode": "TIHA - Brez kazalnikov na uporabniškem vmesniku",
                "language_support": "Slovenščina + angleščina",
                "core_functions": {
                    "promotion_finder": "Iskanje akcijskih izdelkov (tiho validirano)",
                    "item_comparison": "Primerjava cen med trgovinami (tiho validirano)",
                    "meal_search": "Iskanje jedi z analizo (tiho validirano)",
                    "grocery_analysis": "Analiza stroškov nakupovanja",
                    "reverse_meal_search": "Iskanje jedi z vašimi sestavinami (tiho validirano)"
                },
                "silent_validation_features": [
                    f"🎯 Prag relevantnosti: {MIN_RELEVANCE_THRESHOLD}%",
                    "🚫 Avtomatsko filtriranje slabih rezultatov",
                    "📊 Inteligentno razvrščanje po relevantnosti",
                    "📈 Notranje spremljanje kakovosti",
                    "🔍 Brez kazalnikov validacije na uporabniškem vmesniku"
                ],
                "database_status": "povezan" if db_handler else "ni povezan",
                "stores_supported": ["DM", "Lidl", "Mercator", "SPAR", "Tuš"],
                "meal_apis": ["Spoonacular", "Edamam", "TheMealDB"],
                "languages_supported": ["slovenščina", "angleščina"]
            },
            message="Sistem deluje s TIHO validacijo relevantnosti in slovensko podporo",
            approach="sistem_tihe_validacije"
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            error=str(e),
            message="Preverjanje stanja sistema ni uspelo"
        )

@app.get("/")
async def root():
    """Osnovni endpoint z informacijami o slovenski podpori"""
    return {
        "message": "🛒 Sistem za pametno nakupovanje v Sloveniji v3.0",
        "architecture": "Tiha validacija relevantnosti - Brez kazalnikov na uporabniškem vmesniku",
        "validation_mode": "BACKEND ONLY - AI validira in filtrira interno",
        "language_support": "🇸🇮 Slovenščina + 🇬🇧 angleščina",
        "core_functions": [
            {
                "name": "Inteligentno obdelovanje zahtev",
                "endpoint": "/api/intelligent-request",
                "description": "Obdelava naravnega jezika s tiho validacijo"
            },
            {
                "name": "Iskanje akcij", 
                "endpoint": "/api/promotions",
                "description": "Iskanje ponudb s tiho filtriranjem relevantnosti"
            },
            {
                "name": "Primerjava cen",
                "endpoint": "/api/compare-prices", 
                "description": "Primerjava cen s tiho validacijo"
            },
            {
                "name": "Iskanje jedi in analiza",
                "endpoints": ["/api/search-meals", "/api/meal-grocery-analysis", "/api/meals-from-ingredients"],
                "description": "Popoln delovni proces jedi s tiho validacijo"
            }
        ],
        "silent_validation_features": [
            f"🎯 Prag relevantnosti {MIN_RELEVANCE_THRESHOLD}%",
            "🚫 Avtomatsko filtriranje slabih rezultatov",
            "📊 Inteligentno razvrščanje",
            "📈 Notranje spremljanje kakovosti",
            "🔍 Brez kazalnikov validacije na uporabniškem vmesniku"
        ],
        "user_experience": "Čisti rezultati brez validacijskih oznak",
        "getting_started": {
            "note": "Vsi končni endpoints vračajo čiste, validirane rezultate",
            "examples": [
                "najdi poceni vegetarijansko mleko",
                "primerjaj cene bio kruha", 
                "zdrava italijanska večerja za 4 osebe",
                "jedi s piščancem in zelenjavo"
            ]
        },
        "supported_stores": ["DM", "Lidl", "Mercator", "SPAR", "Tuš"],
        "database_language": "Slovenščina",
        "response_language": "Slovenščina z angleško podporo"
    }

# Keep all other endpoints with Slovenian message support...
# [Additional endpoints would follow the same pattern with Slovenian messages]

# Error handlers with Slovenian support
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return APIResponse(
        success=False,
        error="Končna točka ni najdena",
        message="Zahtevana končna točka ne obstaja"
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Notranja napaka strežnika: {exc}")
    return APIResponse(
        success=False,
        error="Notranja napaka strežnika",
        message="Prišlo je do nepričakovane napake"
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