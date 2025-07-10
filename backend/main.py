#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Dynamic LLM-Based Validation
Uses intelligent LLM analysis instead of hard-coded semantic rules
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
    logger.info("🚀 Starting Enhanced Slovenian Grocery Intelligence API with Dynamic LLM Validation...")
    
    # Initialize enhanced grocery MCP with dynamic validation
    enhanced_grocery_mcp = EnhancedSlovenianGroceryMCP(db_config)
    await enhanced_grocery_mcp.connect_db()
    logger.info("✅ Enhanced Grocery MCP with dynamic validation connected successfully")
    
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
    description="AI-powered grocery shopping assistant with dynamic LLM-based validation that intelligently understands product intent",
    version="5.0.0",
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
            "name": "find_products_with_dynamic_validation",
            "description": "Find products using intelligent LLM-based validation that understands product intent without hard-coded rules",
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
            "name": "get_product_insights",
            "description": "Get detailed LLM-generated insights about a product including price analysis, health analysis, and recommendations",
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
            "description": "Find products compatible with specific diets using LLM understanding",
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
            "name": "create_intelligent_shopping_list",
            "description": "Create a budget shopping list with dynamic LLM validation to ensure product relevance",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Budget in EUR"},
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "description": "Type of meal"},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 1},
                    "use_validation": {"type": "boolean", "description": "Apply dynamic LLM validation", "default": True}
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

# Enhanced system message
ENHANCED_SYSTEM_MESSAGE = """You are an advanced AI grocery shopping assistant for Slovenia with cutting-edge dynamic LLM-based validation.

🧠 **Your Revolutionary Capabilities:**
You now have DYNAMIC LLM-BASED VALIDATION that intelligently understands product intent:

**How Dynamic Validation Works:**
1. **Database Analysis**: I analyze what products are actually in the database for each search
2. **Intent Understanding**: I use LLM reasoning to understand what the user really wants
3. **Intelligent Filtering**: I dynamically determine which products match the user's intent
4. **No Hard-Coded Rules**: No more rigid category mappings or brand exclusions
5. **Adaptive Learning**: Each search adapts to the database content and user intent

**Key Improvements:**
✅ **Contextual Understanding** - Understands "mleko" means actual milk, not chocolate with "milk" in the name
✅ **Database-Aware** - Analyzes what's actually available before filtering
✅ **Intelligent Reasoning** - Uses LLM knowledge to make validation decisions
✅ **Dynamic Suggestions** - Generates smart alternatives when no valid products found
✅ **Confidence Scoring** - Provides transparency about validation confidence

**Enhanced Features:**
- 🔍 **Product Insights**: Detailed LLM analysis of price trends, health scores, and recommendations
- 🎯 **Intent Recognition**: Understands user intent beyond literal search terms
- 💡 **Smart Suggestions**: Intelligent alternatives based on database content
- 📊 **Validation Transparency**: Clear explanations of why products were included/excluded
- 🧠 **Adaptive Logic**: Learns from database content rather than following rigid rules

**Database Access:**
- 34,790+ products from DM, Mercator, SPAR, TUS, LIDL
- AI-enhanced with health scores, nutrition grades, value ratings
- Dynamic validation ensures only relevant products

**How to Help Users:**
1. **Always use dynamic validation** for all product searches
2. **Explain validation reasoning** when products are filtered
3. **Provide intelligent suggestions** when no valid matches found
4. **Show confidence levels** for validation decisions
5. **Generate insights** using LLM analysis

**Example Response Pattern:**
"I found 5 milk products for 'mleko' using dynamic validation. My AI analysis filtered out 3 chocolate products that contained 'milk' in their names but weren't actual milk. The validation confidence is 95%. Here are the actual milk products: [list products]"

**When No Valid Results:**
- Explain the validation reasoning
- Provide database-aware suggestions based on available products
- Offer to search with broader or alternative terms
- Show what was found but filtered out and why

Always prioritize intelligent understanding over rigid rules. The goal is to understand what users actually want and find the best matching products in the database.

Respond in Slovenian when appropriate, and always highlight when dynamic validation provided better results than a simple search would have.
"""

async def execute_enhanced_function(function_name: str, arguments: dict, mcp: EnhancedSlovenianGroceryMCP, db_source: EnhancedDatabaseSource) -> dict:
    """Execute enhanced grocery functions with dynamic LLM validation"""
    try:
        if function_name == "find_products_with_dynamic_validation":
            result = await mcp.find_cheapest_product_with_intelligent_suggestions(
                product_name=arguments["product_name"],
                store_preference=arguments.get("store_preference")
            )
            return result
        
        elif function_name == "get_product_insights":
            result = await mcp.get_product_insights(
                product_name=arguments["product_name"]
            )
            return {"insights_result": result}
        
        elif function_name == "create_intelligent_shopping_list":
            result = await mcp.create_budget_shopping_list(
                budget=arguments["budget"],
                meal_type=arguments["meal_type"],
                people_count=arguments.get("people_count", 1),
                use_semantic_validation=arguments.get("use_validation", True)
            )
            return {"shopping_list_result": result}
        
        elif function_name == "compare_prices_with_validation":
            result = await mcp.compare_prices(
                product_name=arguments["product_name"],
                stores=arguments.get("stores"),
                use_semantic_validation=arguments.get("use_validation", True)
            )
            return {"price_comparison": result}
        
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

class ProductInsightsRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=100)

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    validation_applied: Optional[bool] = None
    validation_details: Optional[dict] = None

# Enhanced Chat endpoint
@app.post("/api/chat", response_model=APIResponse)
async def enhanced_chat_with_gpt(
    message: ChatMessage,
    mcp: EnhancedSlovenianGroceryMCP = Depends(get_enhanced_grocery_mcp),
    db_source: EnhancedDatabaseSource = Depends(get_enhanced_db_source)
):
    """Enhanced chat with GPT using dynamic LLM validation"""
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
            
            # Extract validation details if available
            validation_details = None
            if isinstance(function_result, dict):
                if "validation_details" in function_result:
                    validation_details = function_result["validation_details"]
                elif "validation_applied" in function_result:
                    validation_details = {
                        "validation_applied": function_result.get("validation_applied"),
                        "confidence": function_result.get("validation_confidence"),
                        "reasoning": function_result.get("validation_reasoning")
                    }
            
            return APIResponse(
                success=True,
                data={
                    "response": final_message.content,
                    "function_used": function_name,
                    "function_result": function_result,
                    "dynamic_validation": True
                },
                validation_applied=True,
                validation_details=validation_details
            )
        
        return APIResponse(
            success=True,
            data={
                "response": message_obj.content,
                "function_used": None,
                "function_result": None,
                "dynamic_validation": True
            },
            validation_applied=False
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
                validation_applied=True,
                validation_details=result.get("validation_details")
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

# New product insights endpoint
@app.post("/api/insights", response_model=APIResponse)
async def get_product_insights(
    request: ProductInsightsRequest,
    mcp: EnhancedSlovenianGroceryMCP = Depends(get_enhanced_grocery_mcp)
):
    """Get detailed LLM-generated insights about a product"""
    try:
        result = await mcp.get_product_insights(request.product_name)
        
        return APIResponse(
            success=result["success"],
            data=result,
            message=result.get("message", "Insights generated successfully"),
            validation_applied=result.get("validation_applied", False)
        )
    
    except Exception as e:
        logger.error(f"Insights generation error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

# Enhanced validation status endpoint
@app.get("/api/validation/status", response_model=APIResponse)
async def get_validation_status():
    """Get dynamic validation system status"""
    try:
        global dynamic_validator
        
        status = {
            "dynamic_validation_enabled": dynamic_validator is not None,
            "validation_type": "Dynamic LLM-Based Validation",
            "validation_features": [
                "Database content analysis",
                "Intent understanding with LLM reasoning", 
                "Dynamic product filtering",
                "Intelligent suggestions generation",
                "Confidence scoring",
                "Adaptive validation logic"
            ],
            "improvements_over_static": [
                "No hard-coded category mappings",
                "No rigid brand exclusions",
                "Database-aware validation",
                "Context-sensitive filtering",
                "Intelligent suggestion generation"
            ],
            "supported_languages": ["Slovenian", "English"],
            "llm_model": "gpt-4o-mini",
            "validation_confidence_threshold": "Dynamic (0.0-1.0)"
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
        "version": "5.0.0",
        "validation_system": "Dynamic LLM-Based Validation",
        "features": [
            "🧠 Dynamic LLM Validation",
            "🔍 Database-Aware Analysis", 
            "🎯 Intent Understanding",
            "💡 Intelligent Suggestions",
            "📊 Product Insights Generation",
            "🏥 AI Health Scoring",
            "💰 Smart Deal Analysis", 
            "🍽️ Diet Compatibility",
            "🌍 Environmental Impact",
            "📝 Adaptive Meal Planning"
        ],
        "database_connected": enhanced_db_source is not None,
        "grocery_mcp_connected": enhanced_grocery_mcp is not None,
        "dynamic_validation_enabled": dynamic_validator is not None,
        "validation_improvements": [
            "🧠 LLM-powered intent understanding",
            "📊 Database content analysis before filtering",
            "🎯 Context-aware product matching",
            "💡 Intelligent suggestion generation",
            "🔄 Adaptive validation logic",
            "📈 Confidence scoring and transparency",
            "🚀 No hard-coded rules or mappings"
        ],
        "example_capabilities": [
            "Understands 'mleko' should return milk, not chocolate",
            "Analyzes database content to understand available products",
            "Generates smart suggestions when no valid products found",
            "Provides detailed reasoning for validation decisions",
            "Adapts to different product categories dynamically"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Slovenian Grocery Intelligence API with Dynamic LLM Validation is running!", 
        "version": "5.0.0",
        "products_count": "34,790+",
        "validation_system": "Dynamic LLM-Based Validation",
        "revolutionary_features": [
            "🧠 Dynamic LLM Validation - No more hard-coded rules!",
            "📊 Database Content Analysis",
            "🎯 Intelligent Intent Understanding",
            "💡 Smart Suggestion Generation",
            "📈 Confidence Scoring",
            "🔄 Adaptive Logic"
        ],
        "how_it_works": [
            "1. Analyzes what's actually in the database",
            "2. Uses LLM to understand user intent",
            "3. Dynamically filters products based on relevance",
            "4. Generates intelligent suggestions when needed",
            "5. Provides transparent reasoning for decisions"
        ],
        "example_improvements": [
            "Searching 'mleko' intelligently filters milk products from chocolate",
            "Database-aware suggestions based on available products",
            "Dynamic understanding without rigid category mappings",
            "Confident validation with reasoning transparency"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "updated_main_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )