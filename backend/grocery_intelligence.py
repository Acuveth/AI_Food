#!/usr/bin/env python3
"""
Enhanced Slovenian Grocery Intelligence with Think-First Approach
First generates intelligent product lists, then searches database
"""

import asyncio
import json
import logging
import pymysql
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from openai import OpenAI

# Import the new intelligent product generator
from intelligent_product_generator import IntelligentProductGenerator

# Import the dynamic validator
from semantic_search_validation import DynamicSemanticValidator, EnhancedProductSearchWithDynamicValidation

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DietType(Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    KETO = "keto"
    MEDITERRANEAN = "mediterranean"
    GLUTEN_FREE = "gluten_free"
    ORGANIC_ONLY = "organic_only"
    ANY = "any"

class MealType(Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    ANY = "any"

class StoreType(Enum):
    DM = "dm"
    LIDL = "lidl"
    MERCATOR = "mercator"
    SPAR = "spar"
    TUS = "tus"

@dataclass
class ProductResult:
    """Represents a product search result"""
    product_name: str
    store_name: str
    current_price: float
    regular_price: Optional[float] = None
    has_discount: bool = False
    discount_percentage: Optional[int] = None
    product_url: Optional[str] = None
    ai_main_category: Optional[str] = None
    ai_subcategory: Optional[str] = None
    ai_confidence: Optional[str] = None
    ai_health_score: Optional[float] = None
    ai_nutrition_grade: Optional[str] = None
    ai_value_rating: Optional[int] = None
    validation_confidence: Optional[float] = None
    generated_match: Optional[bool] = False  # New field to track if it was in generated list

class DatabaseManager:
    """Handles database connections and operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self._connection_pool = []
        self.max_connections = 10
    
    async def connect(self) -> None:
        """Establish database connection"""
        try:
            self.connection = pymysql.connect(**self.config)
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
    
    def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Database connection closed")
    
    def get_cursor(self) -> pymysql.cursors.DictCursor:
        """Get database cursor"""
        if not self.connection:
            raise ConnectionError("Database not connected")
        return self.connection.cursor(pymysql.cursors.DictCursor)
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """Execute a query and return results"""
        cursor = self.get_cursor()
        try:
            cursor.execute(query, params or [])
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    async def execute_single_query(self, query: str, params: Optional[List] = None) -> Optional[Dict]:
        """Execute a query and return single result"""
        results = await self.execute_query(query, params)
        return results[0] if results else None

class EnhancedSlovenianGroceryMCP:
    """Enhanced grocery intelligence system with think-first approach"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_manager = DatabaseManager(db_config)
        self.dynamic_validator = DynamicSemanticValidator()
        self.enhanced_search = None  # Will be initialized after db connection
        self.product_generator = IntelligentProductGenerator()  # New intelligent generator
        self.stores = [store.value for store in StoreType]
        self.meal_templates = {
            "breakfast": ["mleko", "kruh", "jajca", "maslo", "džem", "jogurt"],
            "lunch": ["meso", "riž", "zelenjava", "olje", "sol", "čebula"],
            "dinner": ["riba", "krompir", "solata", "olje", "limona", "paradižnik"],
            "snack": ["sadež", "oreščki", "jogurt", "čokolada"]
        }
    
    async def connect_db(self) -> None:
        """Connect to database"""
        await self.db_manager.connect()
        # Initialize enhanced search after database connection
        self.enhanced_search = EnhancedProductSearchWithDynamicValidation(self, None)
    
    def disconnect_db(self) -> None:
        """Disconnect from database"""
        self.db_manager.disconnect()
    
    # Utility methods
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '':
            return None
        try:
            if isinstance(value, str):
                value = value.replace('€', '').replace(',', '.').strip()
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer"""
        if value is None or value == '':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_bool(self, value: Any) -> bool:
        """Safely convert value to boolean"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def _format_product_result(self, row: Dict) -> Dict:
        """Format database row as product result"""
        return {
            "product_name": row.get("product_name", ""),
            "store_name": row.get("store_name", ""),
            "current_price": self._safe_float(row.get("current_price", 0)) or 0.0,
            "regular_price": self._safe_float(row.get("regular_price")),
            "has_discount": self._safe_bool(row.get("has_discount", False)),
            "discount_percentage": self._safe_int(row.get("discount_percentage")),
            "product_url": row.get("product_url"),
            "ai_main_category": row.get("ai_main_category"),
            "ai_subcategory": row.get("ai_subcategory"),
            "ai_confidence": row.get("ai_confidence"),
            "ai_health_score": self._safe_float(row.get("ai_health_score")),
            "ai_nutrition_grade": row.get("ai_nutrition_grade"),
            "ai_value_rating": self._safe_int(row.get("ai_value_rating")),
            "ai_product_summary": row.get("ai_product_summary"),
            "ai_diet_compatibility": row.get("ai_diet_compatibility")
        }
    
    async def _search_generated_products(self, generated_products: List[Dict], max_results: int = 50) -> List[Dict]:
        """Search for generated products in database"""
        all_found_products = []
        
        for product_info in generated_products:
            product_name = product_info.get("name", "")
            alternatives = product_info.get("alternatives", [])
            
            # Search for main product name
            if product_name:
                search_result = await self.find_cheapest_product(
                    product_name, use_semantic_validation=True
                )
                
                for product in search_result:
                    product["generated_match"] = True
                    product["generated_info"] = product_info
                    all_found_products.append(product)
            
            # Search for alternatives if main search didn't return many results
            if len(all_found_products) < 3:
                for alt_name in alternatives[:2]:  # Limit to 2 alternatives
                    search_result = await self.find_cheapest_product(
                        alt_name, use_semantic_validation=True
                    )
                    
                    for product in search_result:
                        product["generated_match"] = True
                        product["generated_info"] = product_info
                        all_found_products.append(product)
        
        # Remove duplicates based on product name and store
        seen = set()
        unique_products = []
        for product in all_found_products:
            key = (product.get("product_name", ""), product.get("store_name", ""))
            if key not in seen:
                seen.add(key)
                unique_products.append(product)
        
        # Sort by price and limit results
        unique_products.sort(key=lambda x: x.get("current_price", 0))
        
        logger.info(f"🔍 Found {len(unique_products)} products from generated list")
        return unique_products[:max_results]
    
    # Enhanced functions with think-first approach
    async def get_intelligent_health_focused_products(self, min_health_score: int = 7, limit: int = 50) -> Dict[str, Any]:
        """Get health-focused products using think-first approach"""
        try:
            # Step 1: Generate intelligent product list
            logger.info("🧠 Generating intelligent health-focused product list...")
            generated_list = await self.product_generator.generate_health_focused_products()
            
            # Step 2: Search for generated products in database
            logger.info("🔍 Searching for generated products in database...")
            found_products = await self._search_generated_products(
                generated_list.get("products", []), max_results=limit
            )
            
            # Step 3: Call original function for comparison
            logger.info("📊 Calling original function for comparison...")
            try:
                from database_source import EnhancedDatabaseSource, get_database_config
                db_source = EnhancedDatabaseSource(get_database_config())
                await db_source.connect()
                original_results = await db_source.get_health_focused_products(min_health_score, limit)
                await db_source.disconnect()
            except Exception as e:
                logger.warning(f"Original function failed: {e}")
                original_results = []
            
            # Step 4: Combine and analyze results
            combined_results = self._combine_results(found_products, original_results, "health_focused")
            
            return {
                "success": True,
                "products": combined_results,
                "generated_count": len(found_products),
                "original_count": len(original_results),
                "total_products": len(combined_results),
                "approach": "think_first_then_search",
                "generated_list": generated_list,
                "message": f"Found {len(combined_results)} health-focused products using intelligent generation"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent health search: {e}")
            return {
                "success": False,
                "products": [],
                "error": str(e),
                "approach": "think_first_then_search"
            }
    
    async def get_intelligent_diet_compatible_products(self, diet_type: str, limit: int = 50) -> Dict[str, Any]:
        """Get diet-compatible products using think-first approach"""
        try:
            # Step 1: Generate intelligent product list
            logger.info(f"🧠 Generating intelligent {diet_type} product list...")
            generated_list = await self.product_generator.generate_diet_compatible_products(diet_type)
            
            # Step 2: Search for generated products in database
            logger.info("🔍 Searching for generated products in database...")
            found_products = await self._search_generated_products(
                generated_list.get("products", []), max_results=limit
            )
            
            # Step 3: Call original function for comparison
            logger.info("📊 Calling original function for comparison...")
            try:
                from database_source import EnhancedDatabaseSource, get_database_config
                db_source = EnhancedDatabaseSource(get_database_config())
                await db_source.connect()
                original_results = await db_source.get_diet_compatible_products(diet_type, limit)
                await db_source.disconnect()
            except Exception as e:
                logger.warning(f"Original function failed: {e}")
                original_results = []
            
            # Step 4: Combine and analyze results
            combined_results = self._combine_results(found_products, original_results, "diet_compatible")
            
            return {
                "success": True,
                "products": combined_results,
                "diet_type": diet_type,
                "generated_count": len(found_products),
                "original_count": len(original_results),
                "total_products": len(combined_results),
                "approach": "think_first_then_search",
                "generated_list": generated_list,
                "message": f"Found {len(combined_results)} {diet_type} products using intelligent generation"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent diet search: {e}")
            return {
                "success": False,
                "products": [],
                "error": str(e),
                "approach": "think_first_then_search"
            }
    
    async def get_intelligent_meal_planning_suggestions(self, meal_type: str, people_count: int = 1, budget: float = None) -> Dict[str, Any]:
        """Get meal planning suggestions using think-first approach"""
        try:
            # Step 1: Generate intelligent product list
            logger.info(f"🧠 Generating intelligent meal planning for {meal_type}...")
            generated_list = await self.product_generator.generate_meal_planning_products(meal_type, people_count, budget)
            
            # Step 2: Search for generated products in database
            logger.info("🔍 Searching for generated products in database...")
            found_products = await self._search_generated_products(
                generated_list.get("products", []), max_results=30
            )
            
            # Step 3: Call original function for comparison
            logger.info("📊 Calling original function for comparison...")
            try:
                from database_source import EnhancedDatabaseSource, get_database_config
                db_source = EnhancedDatabaseSource(get_database_config())
                await db_source.connect()
                original_results = await db_source.get_meal_planning_suggestions(meal_type, "moderate")
                await db_source.disconnect()
            except Exception as e:
                logger.warning(f"Original function failed: {e}")
                original_results = []
            
            # Step 4: Combine and analyze results
            combined_results = self._combine_results(found_products, original_results, "meal_planning")
            
            # Step 5: Calculate shopping list with budget
            if budget:
                shopping_list = self._create_optimized_shopping_list(combined_results, budget, people_count)
            else:
                shopping_list = combined_results
            
            return {
                "success": True,
                "products": shopping_list,
                "meal_type": meal_type,
                "people_count": people_count,
                "budget": budget,
                "generated_count": len(found_products),
                "original_count": len(original_results),
                "total_products": len(combined_results),
                "approach": "think_first_then_search",
                "generated_list": generated_list,
                "message": f"Found {len(combined_results)} products for {meal_type} meal planning"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent meal planning: {e}")
            return {
                "success": False,
                "products": [],
                "error": str(e),
                "approach": "think_first_then_search"
            }
    
    async def get_intelligent_seasonal_recommendations(self, season: str = None) -> Dict[str, Any]:
        """Get seasonal recommendations using think-first approach"""
        try:
            # Step 1: Generate intelligent product list
            logger.info(f"🧠 Generating intelligent seasonal products for {season}...")
            generated_list = await self.product_generator.generate_seasonal_products(season)
            
            # Step 2: Search for generated products in database
            logger.info("🔍 Searching for generated products in database...")
            found_products = await self._search_generated_products(
                generated_list.get("products", []), max_results=40
            )
            
            # Step 3: Call original function for comparison
            logger.info("📊 Calling original function for comparison...")
            try:
                from database_source import EnhancedDatabaseSource, get_database_config
                db_source = EnhancedDatabaseSource(get_database_config())
                await db_source.connect()
                original_results = await db_source.get_seasonal_recommendations(season)
                await db_source.disconnect()
            except Exception as e:
                logger.warning(f"Original function failed: {e}")
                original_results = []
            
            # Step 4: Combine and analyze results
            combined_results = self._combine_results(found_products, original_results, "seasonal")
            
            return {
                "success": True,
                "products": combined_results,
                "season": season or "current",
                "generated_count": len(found_products),
                "original_count": len(original_results),
                "total_products": len(combined_results),
                "approach": "think_first_then_search",
                "generated_list": generated_list,
                "message": f"Found {len(combined_results)} seasonal products using intelligent generation"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent seasonal search: {e}")
            return {
                "success": False,
                "products": [],
                "error": str(e),
                "approach": "think_first_then_search"
            }
    
    async def get_intelligent_smart_shopping_deals(self, min_deal_quality: str = "good") -> Dict[str, Any]:
        """Get smart shopping deals using think-first approach"""
        try:
            # Step 1: Generate intelligent product list
            logger.info("🧠 Generating intelligent smart shopping product list...")
            generated_list = await self.product_generator.generate_smart_shopping_products()
            
            # Step 2: Search for generated products in database
            logger.info("🔍 Searching for generated products in database...")
            found_products = await self._search_generated_products(
                generated_list.get("products", []), max_results=50
            )
            
            # Step 3: Call original function for comparison
            logger.info("📊 Calling original function for comparison...")
            try:
                from database_source import EnhancedDatabaseSource, get_database_config
                db_source = EnhancedDatabaseSource(get_database_config())
                await db_source.connect()
                original_results = await db_source.get_smart_shopping_deals(min_deal_quality)
                await db_source.disconnect()
            except Exception as e:
                logger.warning(f"Original function failed: {e}")
                original_results = []
            
            # Step 4: Combine and analyze results, prioritizing deals
            combined_results = self._combine_results(found_products, original_results, "smart_deals")
            
            # Step 5: Filter for actual deals and good value
            deal_products = [p for p in combined_results if p.get("has_discount", False)]
            
            return {
                "success": True,
                "products": deal_products,
                "all_products": combined_results,
                "generated_count": len(found_products),
                "original_count": len(original_results),
                "deals_count": len(deal_products),
                "total_products": len(combined_results),
                "approach": "think_first_then_search",
                "generated_list": generated_list,
                "message": f"Found {len(deal_products)} smart shopping deals using intelligent generation"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent smart shopping: {e}")
            return {
                "success": False,
                "products": [],
                "error": str(e),
                "approach": "think_first_then_search"
            }
    
    async def get_intelligent_allergen_safe_products(self, avoid_allergens: List[str]) -> Dict[str, Any]:
        """Get allergen-safe products using think-first approach"""
        try:
            # Step 1: Generate intelligent product list
            logger.info(f"🧠 Generating intelligent allergen-safe products (avoiding {avoid_allergens})...")
            generated_list = await self.product_generator.generate_allergen_safe_products(avoid_allergens)
            
            # Step 2: Search for generated products in database
            logger.info("🔍 Searching for generated products in database...")
            found_products = await self._search_generated_products(
                generated_list.get("products", []), max_results=50
            )
            
            # Step 3: Call original function for comparison
            logger.info("📊 Calling original function for comparison...")
            try:
                from database_source import EnhancedDatabaseSource, get_database_config
                db_source = EnhancedDatabaseSource(get_database_config())
                await db_source.connect()
                original_results = await db_source.get_allergen_safe_products(avoid_allergens)
                await db_source.disconnect()
            except Exception as e:
                logger.warning(f"Original function failed: {e}")
                original_results = []
            
            # Step 4: Combine and analyze results
            combined_results = self._combine_results(found_products, original_results, "allergen_safe")
            
            return {
                "success": True,
                "products": combined_results,
                "avoided_allergens": avoid_allergens,
                "generated_count": len(found_products),
                "original_count": len(original_results),
                "total_products": len(combined_results),
                "approach": "think_first_then_search",
                "generated_list": generated_list,
                "message": f"Found {len(combined_results)} allergen-safe products using intelligent generation"
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent allergen search: {e}")
            return {
                "success": False,
                "products": [],
                "error": str(e),
                "approach": "think_first_then_search"
            }
    
    def _combine_results(self, generated_products: List[Dict], original_products: List[Dict], search_type: str) -> List[Dict]:
        """Combine and deduplicate results from generated and original searches"""
        
        # Start with generated products (they have higher priority)
        combined = []
        seen_products = set()
        
        # Add generated products first
        for product in generated_products:
            key = (product.get("product_name", ""), product.get("store_name", ""))
            if key not in seen_products:
                product["source"] = "generated"
                combined.append(product)
                seen_products.add(key)
        
        # Add original products that we haven't seen
        for product in original_products:
            key = (product.get("product_name", ""), product.get("store_name", ""))
            if key not in seen_products:
                product["source"] = "original"
                combined.append(product)
                seen_products.add(key)
        
        # Sort by relevance (generated first, then by price)
        combined.sort(key=lambda x: (
            0 if x.get("source") == "generated" else 1,
            x.get("current_price", 0)
        ))
        
        logger.info(f"📊 Combined results: {len(generated_products)} generated + {len(original_products)} original = {len(combined)} total")
        return combined
    
    def _create_optimized_shopping_list(self, products: List[Dict], budget: float, people_count: int) -> List[Dict]:
        """Create optimized shopping list within budget"""
        
        # Sort by value (price per quality/health score)
        def calculate_value_score(product):
            price = product.get("current_price", 0)
            health_score = product.get("ai_health_score", 5)
            discount = product.get("discount_percentage", 0)
            
            # Higher health score and discount = better value
            value = (health_score * (100 + discount)) / max(price, 0.01)
            return value
        
        products.sort(key=calculate_value_score, reverse=True)
        
        # Select products within budget
        shopping_list = []
        total_cost = 0.0
        
        for product in products:
            item_cost = product.get("current_price", 0) * people_count
            if total_cost + item_cost <= budget:
                product["quantity"] = people_count
                product["total_cost"] = item_cost
                shopping_list.append(product)
                total_cost += item_cost
        
        return shopping_list
    
    # Keep existing methods for basic functionality
    async def find_cheapest_product(
        self, 
        product_name: str, 
        location: str = "Ljubljana", 
        store_preference: Optional[str] = None,
        use_semantic_validation: bool = True
    ) -> List[Dict]:
        """Find cheapest product across stores with dynamic LLM validation"""
        
        if use_semantic_validation and self.enhanced_search:
            # Use the new dynamic validation approach
            result = await self.enhanced_search.search_products_with_intelligent_validation(
                product_name, max_results=50, validation_enabled=True
            )
            
            if result["success"]:
                logger.info(f"🎯 Dynamic validation success for '{product_name}'")
                return result["products"]
            else:
                logger.info(f"❌ Dynamic validation found no valid products for '{product_name}'")
                return []
        
        # Fallback to raw database search
        return await self._get_raw_product_search(product_name, store_preference, limit=50)

    async def find_cheapest_product_with_intelligent_suggestions(
        self, 
        product_name: str, 
        location: str = "Ljubljana", 
        store_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced search with intelligent LLM-based suggestions"""
        
        if not self.enhanced_search:
            return {
                "success": False,
                "products": [],
                "message": "Enhanced search not available",
                "search_term": product_name,
                "suggestions": []
            }
        
        # Use dynamic validation
        result = await self.enhanced_search.search_products_with_intelligent_validation(
            product_name, max_results=50, validation_enabled=True
        )
        
        # Add additional context for successful results
        if result["success"]:
            result["message"] = f"Found {len(result['products'])} products for '{product_name}' using AI validation"
            if result.get("validation_reasoning"):
                result["validation_details"] = {
                    "reasoning": result["validation_reasoning"],
                    "confidence": result.get("validation_confidence", 0.0),
                    "invalid_products_filtered": result.get("invalid_products_count", 0)
                }
        
        return result

    async def _get_raw_product_search(
        self, 
        product_name: str, 
        store_preference: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict]:
        """Get raw search results from database without validation"""
        try:
            # Enhanced query that gets comprehensive product data
            query = """
            SELECT store_name, product_name, current_price, regular_price, 
                   has_discount, discount_percentage, product_url, 
                   ai_main_category, ai_subcategory, ai_confidence,
                   ai_health_score, ai_nutrition_grade, ai_value_rating,
                   ai_product_summary, ai_diet_compatibility
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND current_price > 0
            """
            params = [f"%{product_name}%"]
            
            if store_preference:
                query += " AND store_name = %s"
                params.append(store_preference)
            
            query += f" ORDER BY current_price ASC LIMIT {limit}"
            
            results = await self.db_manager.execute_query(query, params)
            formatted_results = [self._format_product_result(row) for row in results]
            
            logger.info(f"🔍 Raw search found {len(formatted_results)} products for '{product_name}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in raw product search: {e}")
            return []

if __name__ == "__main__":
    asyncio.run()