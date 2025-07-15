#!/usr/bin/env python3
"""
SLOVENIAN BACKEND with LLM Output Validator
Enhanced with simple LLM-based validation for all responses
NO OLD PRODUCT EVALUATOR REFERENCES
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

# Import ONLY the new LLM validator
from product_output_evaluator import validate_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validation settings
ENABLE_LLM_VALIDATION = True

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

# Pydantic models
class UserInputRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=500)

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

async def llm_validate_response(
    user_input: str, 
    standard_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    LLM VALIDATION: Use LLM to validate and improve all responses
    """
    if not ENABLE_LLM_VALIDATION:
        return standard_result
    
    try:
        logger.info(f"🤖 LLM validating response for: '{user_input[:50]}...'")
        
        print(standard_result)
        # Use the LLM validator
        validated_result = await validate_output(user_input, standard_result)
       
        
        # Log if any changes were made
        if validated_result != standard_result:
            print(validated_result)
            logger.info(f"🔧 LLM improved the response")
        else:
            logger.info(f"✅ LLM approved response as-is")
        
        return validated_result
        
    except Exception as e:
        logger.error(f"❌ LLM validation failed: {e}")
        # Return original result if validation fails
        return standard_result

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Zagon sistema z LLM validatorjem...")
    
    try:
        db_handler = await get_db_handler()
        if db_handler:
            logger.info("✅ Povezava z bazo podatkov vzpostavljena")
            logger.info("✅ LLM validator inicializiran")
            logger.info(f"✅ LLM validacija: {'OMOGOČENA' if ENABLE_LLM_VALIDATION else 'ONEMOGOČENA'}")
    except Exception as e:
        logger.error(f"❌ Neuspešna inicializacija sistema: {e}")
        raise
    
    yield
    
    logger.info("🔄 Zaustavitev sistema...")
    await close_db_handler()
    logger.info("✅ Zaustavitev dokončana")

# Create FastAPI app
app = FastAPI(
    title="Sistem za pametno nakupovanje - LLM Validator",
    description="AI-powered grocery shopping with LLM-based output validation",
    version="4.0.0-llm-validator",
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

# MAIN INTELLIGENT ENDPOINT - WITH LLM VALIDATION
@app.post("/api/intelligent-request", response_model=APIResponse)
async def intelligent_request(request: UserInputRequest):
    """
    GLAVNI ENDPOINT: Vsi delovni procesi z LLM validacijo
    """
    try:
        logger.info(f"🧠 Obdelava zahteve: '{request.input}'")
        
        # Step 1: Interpret user input with Slovenian support
        interpretation = await interpret_user_input(request.input)
        intent = interpretation.get("intent")
        entities = interpretation.get("extracted_entities", {})
        
        logger.info(f"🎯 Namen: {intent}")
        
        # Step 2: Route to appropriate function
        standard_result = None
        
        if intent == "FIND_PROMOTIONS":
            search_term = entities.get("search_term") or (entities.get("items", [None])[0])
            
            standard_result = await find_promotions(
                search_filter=search_term,
                category_filter=entities.get("category"),
                store_filter=entities.get("store_preference"),
                min_discount=entities.get("min_discount"),
                max_price=entities.get("max_price")
            )
        
        elif intent == "COMPARE_ITEM_PRICES":
            item_name = entities.get("search_term") or (entities.get("items", [""])[0])
            if not item_name:
                return APIResponse(
                    success=False,
                    error=get_slovenian_message("no_items"),
                    intent=intent
                )
            
            standard_result = await compare_item_prices(
                item_name=item_name,
                include_similar=True,
                max_results_per_store=5
            )
        
        elif intent == "SEARCH_MEALS":
            standard_result = await search_meals(
                user_request=request.input,
                max_results=20
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
                standard_result = await reverse_meal_search(
                    available_ingredients=ingredients,
                    max_results=10
                )
            except ImportError:
                logger.warning("reverse_meal_search funkcija ni najdena, uporaba nadomestka")
                standard_result = {
                    "success": True,
                    "suggested_meals": [],
                    "available_ingredients": ingredients,
                    "summary": "Iskanje jedi z vašimi sestavinami trenutno ni na voljo"
                }
        
        else:
            # General response - no validation needed for simple responses
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
        
        # Step 3: LLM VALIDATE AND IMPROVE THE RESULT
        if standard_result:
            validated_result = await llm_validate_response(request.input, standard_result)
            
            return APIResponse(
                success=validated_result["success"],
                data=validated_result,
                message=validated_result.get("summary", validated_result.get("message", "Rezultat obdelan")),
                intent=intent,
                approach=f"{intent.lower()}_llm_validated"
            )
        else:
            return APIResponse(
                success=False,
                error="Ni bilo mogoče generirati rezultata",
                intent=intent
            )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri obdelavi zahteve: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message=get_slovenian_message("error")
        )

@app.get("/api/promotions/all")
async def get_promotions_endpoint(search: Optional[str] = Query(None)):
    """
    Direktni endpoint za akcije (kompatibilnost z frontend)
    """
    try:
        search_query = search or "najdi akcije"
        if search:
            search_query = f"najdi akcije {search}"
        
        logger.info(f"🏷️ Direct promotions request: {search_query}")
        print(f"\n🏷️ PROMOTIONS ENDPOINT REQUEST: '{search_query}'")
        
        # Get standard result
        standard_result = await find_promotions(search_filter=search)
        
        # LLM validate with debug output
        validated_result = await llm_validate_response(search_query, standard_result)
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Akcije najdene"),
            approach="promotions_direct_llm_validated"
        )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri iskanju akcij: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Napaka pri iskanju akcij"
        )

@app.get("/api/compare-prices/{item_name}")
async def compare_prices_endpoint(item_name: str):
    """
    Direktni endpoint za primerjavo cen (kompatibilnost z frontend)
    """
    try:
        search_query = f"primerjaj cene {item_name}"
        
        logger.info(f"🔍 Direct price comparison request: {search_query}")
        print(f"\n🔍 PRICE COMPARISON ENDPOINT REQUEST: '{search_query}'")
        
        # Get standard result
        standard_result = await compare_item_prices(item_name=item_name)
        
        # LLM validate with debug output
        validated_result = await llm_validate_response(search_query, standard_result)
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Primerjava cen dokončana"),
            approach="price_comparison_direct_llm_validated"
        )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri primerjavi cen: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Napaka pri primerjavi cen"
        )

@app.post("/api/search-meals")
async def search_meals_endpoint(request: dict):
    """
    Direktni endpoint za iskanje jedi (kompatibilnost z frontend)
    """
    try:
        user_request = request.get("request", "")
        if not user_request:
            return APIResponse(
                success=False,
                error="Ni zahteve za iskanje jedi",
                message="Potrebna je zahteva za iskanje jedi"
            )
        
        logger.info(f"🍽️ Direct meal search request: {user_request}")
        print(f"\n🍽️ MEAL SEARCH ENDPOINT REQUEST: '{user_request}'")
        
        # Get standard result
        standard_result = await search_meals(user_request=user_request)
        
        # LLM validate with debug output
        validated_result = await llm_validate_response(user_request, standard_result)
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Iskanje jedi dokončano"),
            approach="meal_search_direct_llm_validated"
        )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri iskanju jedi: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Napaka pri iskanju jedi"
        )

@app.post("/api/meals-from-ingredients")
async def meals_from_ingredients_endpoint(request: dict):
    """
    Direktni endpoint za iskanje jedi iz sestavin (kompatibilnost z frontend)
    """
    try:
        ingredients = request.get("ingredients", [])
        if not ingredients:
            return APIResponse(
                success=False,
                error="Ni sestavin",
                message="Potrebne so sestavine za iskanje jedi"
            )
        
        # Create a natural language query
        ingredients_str = ", ".join(ingredients)
        search_query = f"kaj lahko skuham z {ingredients_str}"
        
        logger.info(f"🥗 Direct reverse meal search request: {search_query}")
        print(f"\n🥗 REVERSE MEAL SEARCH ENDPOINT REQUEST: '{search_query}'")
        
        # Get standard result
        standard_result = await reverse_meal_search(available_ingredients=ingredients)
        
        # LLM validate with debug output
        validated_result = await llm_validate_response(search_query, standard_result)
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Iskanje jedi iz sestavin dokončano"),
            approach="reverse_meal_search_direct_llm_validated"
        )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri iskanju jedi iz sestavin: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Napaka pri iskanju jedi iz sestavin"
        )


# MEAL GROCERY ANALYSIS ENDPOINT
@app.post("/api/meal-grocery-analysis")
async def meal_grocery_analysis(request: dict):
    """
    Analiza stroškov nakupovanja za specifično jed z LLM validacijo
    """
    try:
        meal_data = request.get("meal_data")
        if not meal_data:
            return APIResponse(
                success=False,
                error="Ni podatkov o jedi",
                message="Potrebni so podatki o jedi za analizo stroškov"
            )
        
        meal_title = meal_data.get('title', 'Unknown')
        logger.info(f"🛒 Analiza stroškov za jed: {meal_title}")
        print(f"\n🛒 MEAL GROCERY ANALYSIS REQUEST: '{meal_title}'")
        
        # Get grocery analysis
        standard_result = await get_meal_with_grocery_analysis(meal_data)
        
        # LLM validate the result with debug output
        validated_result = await llm_validate_response(
            f"grocery analysis for {meal_title}", 
            standard_result
        )
        
        return APIResponse(
            success=validated_result["success"],
            data=validated_result,
            message=validated_result.get("summary", "Analiza stroškov dokončana"),
            approach="meal_grocery_analysis_llm_validated"
        )
        
    except Exception as e:
        logger.error(f"❌ Napaka pri analizi stroškov: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            message="Napaka pri analizi stroškov nakupovanja"
        )


# HEALTH CHECK
@app.get("/api/health")
async def health_check():
    """Health check z LLM validator informacijami"""
    try:
        db_handler = await get_db_handler()
        
        return {
            "status": "zdrav",
            "timestamp": datetime.now(),
            "version": "4.0.0-llm-validator",
            "architecture": "llm_osnovana_validacija",
            "database_connected": db_handler is not None,
            "llm_validation": {
                "enabled": ENABLE_LLM_VALIDATION,
                "description": "LLM analizira vse odgovore in jih izboljša",
                "coverage": "100% - VSI končni resultati LLM validirani",
                "approach": "Enostaven LLM pristop brez slovarjev"
            },
            "modules": [
                "input_interpreter (slovenska podpora)",
                "promotion_finder (LLM validiran)", 
                "item_finder (LLM validiran)",
                "meal_search (LLM validiran)",
                "database_handler (slovenska podpora)",
                "llm_output_validator (nova arhitektura)"
            ]
        }
        
    except Exception as e:
        return {
            "status": "nezdrav",
            "error": str(e),
            "timestamp": datetime.now()
        }

@app.get("/")
async def root():
    """Osnovni endpoint z informacijami o LLM validaciji"""
    return {
        "message": "🛒 Sistem za pametno nakupovanje v Sloveniji v4.0",
        "architecture": "LLM-osnovana validacija - Brez slovarjev",
        "validation_mode": "LLM analizira in izboljša vse odgovore",
        "language_support": "🇸🇮 Slovenščina + 🇬🇧 angleščina"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "streamlined_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )