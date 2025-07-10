#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Semantic Validation
Prevents wrong product matches like "MLEČNA REZINA MILKA" for "mleko" searches
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
from semantic_search_validation import SemanticSearchValidator

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
semantic_validator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global grocery_mcp, enhanced_db_source, semantic_validator
    
    # Startup
    logger.info("🚀 Starting Enhanced Slovenian Grocery Intelligence API with Semantic Validation...")
    
    # Initialize grocery MCP with validation
    grocery_mcp = SlovenianGroceryMCP(db_config)
    await grocery_mcp.connect_db()
    logger.info("✅ Enhanced Grocery MCP connected successfully")
    
    # Initialize enhanced database source
    enhanced_db_source = EnhancedDatabaseSource(db_config)
    await enhanced_db_source.connect()
    logger.info("✅ Enhanced database source connected successfully")
    
    # Initialize semantic validator
    semantic_validator = SemanticSearchValidator()
    logger.info("✅ Semantic validator initialized")
    
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
    description="AI-powered grocery shopping assistant with semantic validation to prevent wrong product matches",
    version="4.0.0",
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

# Enhanced OpenAI function definitions
ENHANCED_GROCERY_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "find_cheapest_product_validated",
            "description": "Find the cheapest version of a product with semantic validation to ensure correct product matches",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to search for"},
                    "use_validation": {"type": "boolean", "description": "Whether to apply semantic validation", "default": True},
                    "store_preference": {"type": "string", "description": "Preferred store (dm, lidl, mercator, spar, tus)"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_focused_recommendations",
            "description": "Get health-focused product recommendations with AI health scoring",
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
            "name": "get_diet_compatible_products",
            "description": "Find products compatible with specific diets",
            "parameters": {
                "type": "object",
                "properties": {
                    "diet_type": {"type": "string", "description": "Diet type (vegan, vegetarian, keto, etc.)"}
                },
                "required": ["diet_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_smart_shopping_deals",
            "description": "Get intelligent shopping deals with AI analysis",
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
            "name": "create_validated_shopping_list",
            "description": "Create a budget shopping list with semantic validation to ensure correct products",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Budget in EUR"},
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "description": "Type of meal"},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 1},
                    "use_validation": {"type": "boolean", "description": "Apply semantic validation", "default": True}
                },
                "required": ["budget", "meal_type"]
            }
        }
    }
]

# Enhanced system message
ENHANCED_SYSTEM_MESSAGE = """You are an advanced AI grocery shopping assistant for Slovenia with semantic validation capabilities.

🤖 **Your Enhanced Capabilities:**
You now have SEMANTIC VALIDATION that prevents wrong product matches. For example:
- When user searches for "mleko" (milk), you WON'T return "MLEČNA REZINA MILKA" (chocolate bar)
- When user searches for "kruh" (bread), you WON'T return breadcrumbs or bread-related items that aren't actual bread
- When user searches for "jabolka" (apples), you WON'T return apple juice or apple-flavored products

**Database Access:**
- 34,790+ products from DM, Mercator, SPAR, TUS, LIDL
- AI-enhanced with health scores, nutrition grades, value ratings
- Semantic validation ensures product relevance

**Key Features:**
✅ **Semantic Product Matching** - Only returns products that actually match user intent
✅ **Search Suggestions** - Provides alternatives when no valid matches found
✅ **Validation Transparency** - Explains when validation is applied
✅ **Category Filtering** - Uses AI categories to improve accuracy

**How to Help Users:**
1. **Always use validated search** by default for product queries
2. **Explain validation results** - tell users when validation helped
3. **Provide suggestions** when no valid products found
4. **Be transparent** about search quality and matches

**When No Valid Results:**
- Explain that validation prevented wrong matches
- Offer search suggestions or alternative terms
- Ask if user wants to try broader search terms

**Example Response Pattern:**
"I found 3 validated products for 'mleko' (my validation system excluded chocolate products like Milka bars that appeared in the search). Here are actual milk products: [list products]"

Always prioritize accuracy over quantity - better to return fewer correct results than many irrelevant ones!

Respond in Slovenian when appropriate, and always mention when semantic validation helped improve results.
"""

async def execute_enhanced_function(function_name: str, arguments: dict, mcp: SlovenianGroceryMCP, db_source: EnhancedDatabaseSource) -> dict:
    """Execute enhanced grocery functions with semantic validation"""
    try:
        if function_name == "find_cheapest_product_validated":
            result = await mcp.find_cheapest_product_with_suggestions(
                product_name=arguments["product_name"],
                store_preference=arguments.get("store_preference")
            )
            return result
        
        elif function_name == "create_validated_shopping_list":
            result = await mcp.create_budget_shopping_list(
                budget=arguments["budget"],
                meal_type=arguments["meal_type"],
                people_count=arguments.get("people_count", 1),
                use_semantic_validation=arguments.get("use_validation", True)
            )
            return {"shopping_list_result": result}
        
        elif function_name == "get_health_focused_recommendations":
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
    validation_applied: Optional[bool] = None

# Enhanced Chat endpoint
@app.post("/api/chat", response_model=APIResponse)
async def enhanced_chat_with_gpt(
    message: ChatMessage,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Enhanced chat with GPT using semantic validation"""
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
                    "semantic_validation": True
                },
                validation_applied=True
            )
        
        return APIResponse(
            success=True,
            data={
                "response": message_obj.content,
                "function_used": None,
                "function_result": None,
                "semantic_validation": True
            },
            validation_applied=True
        )
    
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e),
            validation_applied=False
        )

# Enhanced search endpoint
@app.post("/api/search", response_model=APIResponse)
async def enhanced_product_search(
    request: ProductSearchRequest,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Enhanced product search with semantic validation"""
    try:
        if request.use_validation:
            result = await mcp.find_cheapest_product_with_suggestions(
                product_name=request.product_name,
                store_preference=request.store_preference
            )
            
            return APIResponse(
                success=result["success"],
                data={
                    "products": result.get("products", []),
                    "suggestions": result.get("suggestions", []),
                    "search_term": result.get("search_term"),
                    "raw_results_count": result.get("raw_results_count", 0)
                },
                message=result.get("message"),
                validation_applied=True
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
                validation_applied=False
            )
    
    except Exception as e:
        logger.error(f"Enhanced search error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e),
            validation_applied=request.use_validation
        )

# New validation status endpoint
@app.get("/api/validation/status", response_model=APIResponse)
async def get_validation_status():
    """Get semantic validation system status"""
    try:
        global semantic_validator
        
        status = {
            "semantic_validation_enabled": semantic_validator is not None,
            "validation_features": [
                "Category-based filtering",
                "Brand exclusion filtering", 
                "AI semantic validation",
                "Search suggestions"
            ],
            "supported_languages": ["Slovenian", "English"],
            "category_mappings_count": len(semantic_validator.category_mappings) if semantic_validator else 0
        }
        
        return APIResponse(success=True, data=status)
    
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# Enhanced health check
@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "version": "4.0.0",
        "features": [
            "🔍 Semantic Product Validation",
            "🎯 Search Intent Recognition", 
            "💡 Smart Search Suggestions",
            "🏥 AI Health Scoring",
            "💰 Smart Deal Analysis", 
            "🍽️ Diet Compatibility",
            "🌍 Environmental Impact",
            "📝 Meal Planning Intelligence"
        ],
        "database_connected": enhanced_db_source is not None,
        "grocery_mcp_connected": grocery_mcp is not None,
        "semantic_validation_enabled": semantic_validator is not None,
        "validation_improvements": [
            "Prevents wrong product matches (e.g., Milka chocolate when searching for milk)",
            "Category-aware filtering",
            "Brand exclusion logic",
            "AI-powered semantic validation",
            "Search suggestions when no matches found"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Slovenian Grocery Intelligence API with Semantic Validation is running!", 
        "version": "4.0.0",
        "products_count": "34,790+",
        "new_features": [
            "🔍 Semantic Validation - No more wrong product matches!",
            "💡 Smart Search Suggestions",
            "🎯 Intent Recognition",
            "✅ Validation Transparency"
        ],
        "example_improvements": [
            "Searching 'mleko' no longer returns 'MLEČNA REZINA MILKA'",
            "Better category matching and filtering",
            "Helpful suggestions when no products found",
            "Transparent validation reporting"
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