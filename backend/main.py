#!/usr/bin/env python3
"""
FastAPI Backend for Slovenian Grocery Intelligence
Integrated with real database and grocery intelligence system
"""

import os
import json
import logging
from typing import List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Import the grocery intelligence system
from grocery_intelligence import SlovenianGroceryMCP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_DATABASE', 'ai_food'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'autocommit': True
}

# Global grocery MCP instance
grocery_mcp = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global grocery_mcp
    
    # Startup
    logger.info("🚀 Starting Slovenian Grocery Intelligence API...")
    grocery_mcp = SlovenianGroceryMCP(db_config)
    await grocery_mcp.connect_db()
    logger.info("✅ Database connected successfully")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Slovenian Grocery Intelligence API...")
    if grocery_mcp:
        grocery_mcp.disconnect_db()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Slovenian Grocery Intelligence API",
    description="AI-powered grocery shopping assistant for Slovenia",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    model: str = Field(default="gpt-3.5-turbo")

class ProductSearchRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(default="Ljubljana")
    store_preference: Optional[str] = None

class PriceComparisonRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=100)
    stores: Optional[List[str]] = None

class ShoppingListRequest(BaseModel):
    budget: float = Field(..., gt=0, le=1000)
    meal_type: str = Field(..., pattern="^(breakfast|lunch|dinner|snack)$")
    people_count: int = Field(..., ge=1, le=20)
    dietary_restrictions: Optional[List[str]] = None

class PromotionsRequest(BaseModel):
    store: Optional[str] = None
    category: Optional[str] = None
    min_discount: int = Field(default=10, ge=0, le=100)

class WeeklyMealPlanRequest(BaseModel):
    budget: float = Field(..., gt=0, le=2000)
    people_count: int = Field(..., ge=1, le=20)

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Dependency to get grocery MCP instance
async def get_grocery_mcp() -> SlovenianGroceryMCP:
    """Get grocery MCP instance"""
    if not grocery_mcp:
        raise HTTPException(status_code=500, detail="Grocery system not initialized")
    return grocery_mcp

# OpenAI function definitions
GROCERY_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "find_cheapest_product",
            "description": "Find the cheapest version of a product across all Slovenian stores",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to search for"},
                    "location": {"type": "string", "description": "Location to search in", "default": "Ljubljana"},
                    "store_preference": {"type": "string", "description": "Preferred store (optional)"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_prices",
            "description": "Compare prices of a product across multiple stores",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to compare"},
                    "stores": {"type": "array", "items": {"type": "string"}, "description": "List of stores to compare (optional)"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_budget_shopping_list",
            "description": "Create a budget-optimized shopping list",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Budget in EUR"},
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"]},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 1},
                    "dietary_restrictions": {"type": "array", "items": {"type": "string"}, "description": "Dietary restrictions (optional)"}
                },
                "required": ["budget", "meal_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_promotions",
            "description": "Get current promotions and discounts",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {"type": "string", "description": "Specific store (optional)"},
                    "category": {"type": "string", "description": "Product category (optional)"},
                    "min_discount": {"type": "integer", "description": "Minimum discount percentage", "default": 10}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_availability",
            "description": "Check which stores have a specific product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to check"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ai_insights",
            "description": "Get AI insights and analysis for a product",
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
            "name": "suggest_meal_from_promotions",
            "description": "Suggest a meal based on current promotions",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Budget in EUR"},
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                    "people_count": {"type": "integer", "description": "Number of people", "default": 1}
                },
                "required": ["budget", "meal_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_nutrition_analysis",
            "description": "Get nutrition analysis for a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name to analyze"}
                },
                "required": ["product_name"]
            }
        }
    }
]

async def execute_grocery_function(function_name: str, arguments: dict, mcp: SlovenianGroceryMCP) -> dict:
    """Execute grocery functions with error handling"""
    try:
        if function_name == "find_cheapest_product":
            result = await mcp.find_cheapest_product(
                arguments["product_name"],
                arguments.get("location", "Ljubljana"),
                arguments.get("store_preference")
            )
            return {"products": result}
        
        elif function_name == "compare_prices":
            result = await mcp.compare_prices(
                arguments["product_name"],
                arguments.get("stores")
            )
            return {"comparison": result}
        
        elif function_name == "create_budget_shopping_list":
            result = await mcp.create_budget_shopping_list(
                arguments["budget"],
                arguments["meal_type"],
                arguments.get("people_count", 1),
                arguments.get("dietary_restrictions")
            )
            return {"shopping_list": result}
        
        elif function_name == "get_current_promotions":
            result = await mcp.get_current_promotions(
                arguments.get("store"),
                arguments.get("category"),
                arguments.get("min_discount", 10)
            )
            return {"promotions": result}
        
        elif function_name == "get_store_availability":
            result = await mcp.get_store_availability(arguments["product_name"])
            return {"availability": result}
        
        elif function_name == "get_ai_insights":
            result = await mcp.get_ai_insights(arguments["product_name"])
            return {"insights": result}
        
        elif function_name == "suggest_meal_from_promotions":
            result = await mcp.suggest_meal_from_promotions(
                arguments["budget"],
                arguments["meal_type"],
                arguments.get("people_count", 1)
            )
            return {"meal_suggestion": result}
        
        elif function_name == "get_nutrition_analysis":
            result = await mcp.get_nutrition_analysis(arguments["product_name"])
            return {"nutrition": result}
        
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")

# API Routes
@app.post("/api/chat", response_model=APIResponse)
async def chat_with_gpt(
    message: ChatMessage,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Chat with GPT using grocery functions"""
    try:
        system_message = """You are a helpful Slovenian grocery shopping assistant. You have access to real-time pricing data from major stores in Slovenia: DM, Mercator, SPAR, TUS, and LIDL.

You can help users:
- Find the cheapest products across all stores
- Compare prices between stores
- Create budget-optimized shopping lists
- Find current promotions and discounts
- Check store availability
- Get AI insights about products
- Suggest meals based on promotions
- Get nutrition analysis

Always provide prices in EUR and mention specific stores. Be helpful and focus on saving money. When users ask about products, use the available functions to get current data. Respond in Slovenian when appropriate."""

        # First GPT call
        response = client.chat.completions.create(
            model=message.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message.message}
            ],
            tools=GROCERY_FUNCTIONS,
            tool_choice="auto"
        )
        
        message_obj = response.choices[0].message
        
        # Check if GPT wants to call a function
        if message_obj.tool_calls:
            tool_call = message_obj.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the function
            function_result = await execute_grocery_function(function_name, function_args, mcp)
            
            # Second GPT call with function result
            follow_up_response = client.chat.completions.create(
                model=message.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message.message},
                    {"role": "assistant", "content": message_obj.content, "tool_calls": message_obj.tool_calls},
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_result)}
                ],
                tools=GROCERY_FUNCTIONS,
                tool_choice="auto"
            )
            
            final_message = follow_up_response.choices[0].message
            return APIResponse(
                success=True,
                data={
                    "response": final_message.content,
                    "function_used": function_name,
                    "function_result": function_result
                }
            )
        
        return APIResponse(
            success=True,
            data={
                "response": message_obj.content,
                "function_used": None,
                "function_result": None
            }
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/search", response_model=APIResponse)
async def search_products(
    request: ProductSearchRequest,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Search for products"""
    try:
        results = await mcp.find_cheapest_product(
            request.product_name,
            request.location,
            request.store_preference
        )
        return APIResponse(
            success=True,
            data={"products": results}
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/compare", response_model=APIResponse)
async def compare_prices(
    request: PriceComparisonRequest,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Compare prices across stores"""
    try:
        results = await mcp.compare_prices(request.product_name, request.stores)
        return APIResponse(
            success=True,
            data={"comparison": results}
        )
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/shopping-list", response_model=APIResponse)
async def create_shopping_list(
    request: ShoppingListRequest,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Create budget shopping list"""
    try:
        results = await mcp.create_budget_shopping_list(
            request.budget,
            request.meal_type,
            request.people_count,
            request.dietary_restrictions
        )
        return APIResponse(
            success=True,
            data={"shopping_list": results}
        )
    except Exception as e:
        logger.error(f"Shopping list error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/promotions", response_model=APIResponse)
async def get_promotions(
    request: PromotionsRequest,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Get current promotions"""
    try:
        results = await mcp.get_current_promotions(
            request.store,
            request.category,
            request.min_discount
        )
        return APIResponse(
            success=True,
            data={"promotions": results}
        )
    except Exception as e:
        logger.error(f"Promotions error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/stores/{product_name}", response_model=APIResponse)
async def check_store_availability(
    product_name: str,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Check store availability"""
    try:
        results = await mcp.get_store_availability(product_name)
        return APIResponse(
            success=True,
            data={"availability": results}
        )
    except Exception as e:
        logger.error(f"Store availability error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/insights/{product_name}", response_model=APIResponse)
async def get_product_insights(
    product_name: str,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Get AI insights for product"""
    try:
        results = await mcp.get_ai_insights(product_name)
        return APIResponse(
            success=True,
            data={"insights": results}
        )
    except Exception as e:
        logger.error(f"Insights error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/nutrition/{product_name}", response_model=APIResponse)
async def get_nutrition_analysis(
    product_name: str,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Get nutrition analysis for product"""
    try:
        results = await mcp.get_nutrition_analysis(product_name)
        return APIResponse(
            success=True,
            data={"nutrition": results}
        )
    except Exception as e:
        logger.error(f"Nutrition analysis error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/meal-suggestion", response_model=APIResponse)
async def suggest_meal_from_promotions(
    budget: float,
    meal_type: str,
    people_count: int = 1,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Suggest meal based on promotions"""
    try:
        results = await mcp.suggest_meal_from_promotions(budget, meal_type, people_count)
        return APIResponse(
            success=True,
            data={"meal_suggestion": results}
        )
    except Exception as e:
        logger.error(f"Meal suggestion error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/weekly-meal-plan", response_model=APIResponse)
async def create_weekly_meal_plan(
    request: WeeklyMealPlanRequest,
    mcp: SlovenianGroceryMCP = Depends(get_grocery_mcp)
):
    """Create weekly meal plan"""
    try:
        results = await mcp.get_weekly_meal_plan(request.budget, request.people_count)
        return APIResponse(
            success=True,
            data={"weekly_plan": results}
        )
    except Exception as e:
        logger.error(f"Weekly meal plan error: {str(e)}")
        return APIResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Slovenian Grocery Intelligence API is running!", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )