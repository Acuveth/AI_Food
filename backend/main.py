#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for Slovenian Grocery Intelligence
AI-aware assistant with comprehensive database knowledge
"""

import os
import json
import logging
from typing import List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Import the enhanced systems
from grocery_intelligence import SlovenianGroceryMCP
from database_source import EnhancedDatabaseSource, get_database_config

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
grocery_mcp = None
enhanced_db_source = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global grocery_mcp, enhanced_db_source
    
    # Startup
    logger.info("🚀 Starting Enhanced Slovenian Grocery Intelligence API...")
    
    # Initialize grocery MCP
    grocery_mcp = SlovenianGroceryMCP(db_config)
    await grocery_mcp.connect_db()
    logger.info("✅ Grocery MCP connected successfully")
    
    # Initialize enhanced database source
    enhanced_db_source = EnhancedDatabaseSource(db_config)
    await enhanced_db_source.connect()
    logger.info("✅ Enhanced database source connected successfully")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Enhanced Slovenian Grocery Intelligence API...")
    if grocery_mcp:
        grocery_mcp.disconnect_db()
    if enhanced_db_source:
        enhanced_db_source.disconnect()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Slovenian Grocery Intelligence API",
    description="AI-powered grocery shopping assistant with comprehensive product intelligence",
    version="3.0.0",
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

# Enhanced OpenAI function definitions with AI database features
ENHANCED_GROCERY_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_health_focused_recommendations",
            "description": "Get health-focused product recommendations with AI health scoring, nutrition grades, and dietary analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_health_score": {"type": "integer", "description": "Minimum health score (0-10)", "default": 7},
                    "nutrition_grade": {"type": "string", "description": "Preferred nutrition grade (A, B, C, D, E)", "default": "any"},
                    "max_sugar": {"type": "string", "description": "Maximum sugar level (low, medium, high)", "default": "any"},
                    "max_sodium": {"type": "string", "description": "Maximum sodium level (low, medium, high)", "default": "any"},
                    "organic_only": {"type": "boolean", "description": "Only organic products", "default": False}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_diet_compatible_products",
            "description": "Find products compatible with specific diets using AI dietary analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "diet_type": {"type": "string", "description": "Diet type (vegan, vegetarian, keto, gluten-free, paleo, etc.)"},
                    "avoid_allergens": {"type": "array", "items": {"type": "string"}, "description": "Allergens to avoid"},
                    "min_health_score": {"type": "integer", "description": "Minimum health score", "default": 5}
                },
                "required": ["diet_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_smart_shopping_deals",
            "description": "Get intelligent shopping deals based on AI deal quality analysis, value ratings, and purchase recommendations",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_deal_quality": {"type": "string", "enum": ["excellent", "good", "fair"], "description": "Minimum deal quality", "default": "good"},
                    "stockup_worthy": {"type": "boolean", "description": "Only products worth stocking up on", "default": False},
                    "bulk_discount_worthy": {"type": "boolean", "description": "Only products good for bulk buying", "default": False}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_meal_planning_suggestions",
            "description": "Get intelligent meal planning suggestions with pairing recommendations, preparation tips, and recipe compatibility",
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_category": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "description": "Meal category"},
                    "max_prep_complexity": {"type": "string", "enum": ["simple", "moderate", "complex"], "description": "Maximum preparation complexity", "default": "moderate"},
                    "cuisine_type": {"type": "string", "description": "Cuisine type or recipe style"},
                    "budget_per_person": {"type": "number", "description": "Budget per person in EUR"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_comprehensive_product_analysis",
            "description": "Get comprehensive AI analysis for specific products including health, value, environmental impact, and usage suggestions",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to analyze"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_environmental_recommendations",
            "description": "Get environmentally-friendly product recommendations with sustainability scoring",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_env_score": {"type": "integer", "description": "Minimum environmental score (0-10)", "default": 6},
                    "organic_preferred": {"type": "boolean", "description": "Prefer organic products", "default": True},
                    "minimal_processing": {"type": "boolean", "description": "Prefer minimally processed foods", "default": True}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_recommendations",
            "description": "Get seasonal product recommendations based on freshness and seasonal availability",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "string", "description": "Season (spring, summer, autumn, winter) or current"},
                    "freshness_priority": {"type": "boolean", "description": "Prioritize freshness", "default": True}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_storage_and_usage_tips",
            "description": "Get detailed storage tips, shelf life information, and creative usage suggestions for products",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to get tips for"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_cheapest_product",
            "description": "Find the cheapest version of a product across all stores with AI value analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to search for"},
                    "include_ai_analysis": {"type": "boolean", "description": "Include AI value and quality analysis", "default": True}
                },
                "required": ["product_name"]
            }
        }
    }
]

# Enhanced system message with comprehensive database knowledge
ENHANCED_SYSTEM_MESSAGE = """You are an advanced AI grocery shopping assistant for Slovenia with access to a comprehensive database of 34,790+ products from major stores (DM, Mercator, SPAR, TUS, LIDL). 

🤖 **Your Database Knowledge:**
You have access to incredibly detailed AI-enhanced product data including:

**Health & Nutrition Intelligence:**
- ai_health_score (0-10): Overall health rating for each product
- ai_nutrition_grade (A-E): Traffic light nutrition scoring
- ai_sugar_content, ai_sodium_level: Detailed nutritional analysis
- ai_additive_score: Food additive safety assessment
- ai_processing_level: How processed foods are (minimal/moderate/high)

**Dietary & Allergen Analysis:**
- ai_diet_compatibility: Compatible diets (vegan, vegetarian, keto, gluten-free, etc.)
- ai_allergen_list: Detailed allergen information
- ai_allergen_risk: Risk level assessment (low/medium/high)
- ai_organic_verified: Certified organic status

**Smart Shopping Intelligence:**
- ai_value_rating: Value for money assessment (excellent/good/fair/poor)
- ai_deal_quality: How good current deals are
- ai_price_tier: Price category (budget/mid-range/premium)
- ai_stockup_recommendation: Whether to buy in bulk
- ai_optimal_quantity: Recommended purchase amount
- ai_replacement_urgency: How urgently items need replacing

**Culinary & Usage Intelligence:**
- ai_pairing_suggestions: What foods pair well together
- ai_recipe_compatibility: Recipe types products work in
- ai_preparation_complexity: How complex to prepare (simple/moderate/complex)
- ai_preparation_tips: Detailed cooking/prep instructions
- ai_usage_suggestions: Creative ways to use products
- ai_meal_category: Best meal timing (breakfast/lunch/dinner/snack)

**Storage & Freshness Intelligence:**
- ai_storage_requirements: Optimal storage conditions
- ai_shelf_life_estimate: Expected shelf life in days
- ai_freshness_indicator: Current freshness status
- ai_seasonal_availability: Best seasons for products

**Environmental Intelligence:**
- ai_environmental_score (0-10): Environmental impact rating
- ai_organic_verified: Organic certification status

🎯 **How to Help Users:**

**For Health Queries:** Use health scores, nutrition grades, sugar/sodium levels, and processing information
**For Diet Queries:** Leverage diet compatibility and allergen analysis 
**For Budget Queries:** Use value ratings, deal quality, and smart shopping recommendations
**For Cooking Queries:** Provide pairing suggestions, recipe compatibility, and preparation tips
**For Storage Queries:** Share storage requirements, shelf life, and freshness information
**For Environmental Queries:** Use environmental scores and organic verification

**Always:**
- Provide specific product recommendations with store names and prices in EUR
- Include relevant AI insights (health scores, value ratings, etc.)
- Suggest optimal quantities and storage tips when helpful
- Mention seasonal availability and freshness when relevant
- Highlight great deals and explain why they're good value
- Consider dietary restrictions and allergen safety
- Provide actionable shopping and cooking advice

You're not just finding prices - you're providing intelligent, comprehensive grocery guidance based on deep product analysis!

Respond in Slovenian when appropriate, and always prioritize user health, value, and satisfaction.
"""

async def execute_enhanced_function(function_name: str, arguments: dict, mcp: SlovenianGroceryMCP, db_source: EnhancedDatabaseSource) -> dict:
    """Execute enhanced grocery functions with comprehensive AI analysis"""
    try:
        if function_name == "get_health_focused_recommendations":
            result = await db_source.get_health_focused_products(
                min_health_score=arguments.get("min_health_score", 7)
            )
            return {"health_products": result, "count": len(result)}
        
        elif function_name == "get_diet_compatible_products":
            result = await db_source.get_diet_compatible_products(
                diet_type=arguments["diet_type"]
            )
            return {"diet_products": result, "diet_type": arguments["diet_type"], "count": len(result)}
        
        elif function_name == "get_smart_shopping_deals":
            result = await db_source.get_smart_shopping_deals(
                min_deal_quality=arguments.get("min_deal_quality", "good")
            )
            return {"smart_deals": result, "count": len(result)}
        
        elif function_name == "get_meal_planning_suggestions":
            result = await db_source.get_meal_planning_suggestions(
                meal_category=arguments.get("meal_category"),
                max_prep_complexity=arguments.get("max_prep_complexity", "moderate")
            )
            return {"meal_suggestions": result, "count": len(result)}
        
        elif function_name == "get_comprehensive_product_analysis":
            result = await db_source.get_comprehensive_product_analysis(
                product_name=arguments["product_name"]
            )
            return {"product_analysis": result, "product": arguments["product_name"], "count": len(result)}
        
        elif function_name == "get_environmental_recommendations":
            result = await db_source.get_environmental_impact_analysis(
                min_env_score=arguments.get("min_env_score", 6)
            )
            return {"eco_products": result, "count": len(result)}
        
        elif function_name == "get_seasonal_recommendations":
            result = await db_source.get_seasonal_recommendations(
                season=arguments.get("season")
            )
            return {"seasonal_products": result, "count": len(result)}
        
        elif function_name == "get_storage_and_usage_tips":
            result = await db_source.get_storage_and_freshness_tips(
                product_name=arguments["product_name"]
            )
            return {"storage_tips": result, "product": arguments["product_name"], "count": len(result)}
        
        elif function_name == "find_cheapest_product":
            # Use the original MCP function but enhance with AI data
            result = await mcp.find_cheapest_product(arguments["product_name"])
            
            # Also get AI analysis
            ai_analysis = await db_source.get_comprehensive_product_analysis(arguments["product_name"])
            
            return {
                "products": result, 
                "ai_analysis": ai_analysis[:3],  # Top 3 AI analyses
                "product_name": arguments["product_name"]
            }
        
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")

# Dependencies
async def get_grocery_mcp() -> SlovenianGroceryMCP:
    """Get grocery MCP instance"""
    if not grocery_mcp:
        raise HTTPException(status_code=500, detail="Grocery system not initialized")
    return grocery_mcp

async def get_enhanced_db_source() -> EnhancedDatabaseSource:
    """Get enhanced database source instance"""
    if not enhanced_db_source:
        raise HTTPException(status_code=500, detail="Enhanced database source not initialized")
    return enhanced_db_source

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    model: str = Field(default="gpt-3.5-turbo")

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Enhanced Chat endpoint with AI database features
@app.post("/api/chat", response_model=APIResponse)
async def enhanced_chat_with_gpt(
    message: ChatMessage,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Enhanced chat with GPT using comprehensive AI grocery intelligence"""
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
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_result)}
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
                    "ai_enhanced": True
                }
            )
        
        return APIResponse(
            success=True,
            data={
                "response": message_obj.content,
                "function_used": None,
                "function_result": None,
                "ai_enhanced": True
            }
        )
    
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

# New enhanced endpoints
@app.get("/api/enhanced/health-products", response_model=APIResponse)
async def get_health_products(
    min_health_score: int = Query(default=7, ge=0, le=10),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Get health-focused products"""
    try:
        products = await db_source.get_health_focused_products(min_health_score)
        return APIResponse(success=True, data={"products": products, "count": len(products)})
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/enhanced/diet/{diet_type}", response_model=APIResponse)
async def get_diet_products(
    diet_type: str,
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Get products for specific diet"""
    try:
        products = await db_source.get_diet_compatible_products(diet_type)
        return APIResponse(success=True, data={"products": products, "diet": diet_type, "count": len(products)})
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/enhanced/smart-deals", response_model=APIResponse)
async def get_smart_deals(
    min_quality: str = Query(default="good", regex="^(excellent|good|fair)$"),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Get smart shopping deals"""
    try:
        deals = await db_source.get_smart_shopping_deals(min_quality)
        return APIResponse(success=True, data={"deals": deals, "count": len(deals)})
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/enhanced/environmental", response_model=APIResponse)
async def get_environmental_products(
    min_env_score: int = Query(default=6, ge=0, le=10),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Get environmentally-friendly products"""
    try:
        products = await db_source.get_environmental_impact_analysis(min_env_score)
        return APIResponse(success=True, data={"products": products, "count": len(products)})
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/enhanced/analysis/{product_name}", response_model=APIResponse)
async def get_product_analysis(
    product_name: str,
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Get comprehensive product analysis"""
    try:
        analysis = await db_source.get_comprehensive_product_analysis(product_name)
        return APIResponse(success=True, data={"analysis": analysis, "product": product_name, "count": len(analysis)})
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "version": "3.0.0",
        "features": [
            "AI Health Scoring",
            "Smart Deal Analysis", 
            "Diet Compatibility",
            "Environmental Impact",
            "Meal Planning Intelligence",
            "Storage & Freshness Tips"
        ],
        "database_connected": enhanced_db_source is not None,
        "grocery_mcp_connected": grocery_mcp is not None,
        "ai_enhanced": True
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Slovenian Grocery Intelligence API with AI Analysis is running!", 
        "version": "3.0.0",
        "products_count": "34,790+",
        "ai_features": [
            "Health & Nutrition Scoring",
            "Smart Deal Analysis",
            "Diet Compatibility Checking", 
            "Environmental Impact Assessment",
            "Meal Planning Intelligence",
            "Storage & Freshness Management",
            "Value & Quality Analysis"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )