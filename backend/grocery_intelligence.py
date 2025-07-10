#!/usr/bin/env python3
"""
Enhanced Slovenian Grocery Intelligence with Dynamic LLM-Based Validation
Uses intelligent LLM analysis instead of hard-coded rules
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
    """Enhanced grocery intelligence system with dynamic LLM-based validation"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_manager = DatabaseManager(db_config)
        self.dynamic_validator = DynamicSemanticValidator()
        self.enhanced_search = None  # Will be initialized after db connection
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
    
    # Core functionality methods with dynamic validation
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

    async def compare_prices(
        self, 
        product_name: str, 
        stores: Optional[List[str]] = None,
        use_semantic_validation: bool = True
    ) -> Dict[str, List[Dict]]:
        """Compare prices across stores with dynamic validation"""
        try:
            # Get results with dynamic validation
            if use_semantic_validation and self.enhanced_search:
                result = await self.enhanced_search.search_products_with_intelligent_validation(
                    product_name, max_results=100, validation_enabled=True
                )
                
                if result["success"]:
                    all_results = result["products"]
                    logger.info(f"📊 Using validated results for price comparison")
                else:
                    # Fallback to raw results
                    all_results = await self._get_raw_product_search(product_name, limit=100)
                    logger.info(f"📊 Using raw results for price comparison (validation failed)")
            else:
                # Use raw search
                all_results = await self._get_raw_product_search(product_name, limit=100)
            
            # Filter by stores if specified
            if stores:
                all_results = [r for r in all_results if r['store_name'] in stores]
            
            # Group by store
            store_results = {}
            for row in all_results:
                store = row['store_name']
                if store not in store_results:
                    store_results[store] = []
                store_results[store].append(row)
            
            logger.info(f"📊 Price comparison: {len(store_results)} stores, {len(all_results)} products")
            return store_results
            
        except Exception as e:
            logger.error(f"Error comparing prices: {e}")
            return {}

    async def create_budget_shopping_list(
        self, 
        budget: float, 
        meal_type: str, 
        people_count: int = 1, 
        dietary_restrictions: Optional[List[str]] = None,
        use_semantic_validation: bool = True
    ) -> Dict:
        """Create budget-optimized shopping list with dynamic validation"""
        try:
            # Get base items for meal type
            base_items = self.meal_templates.get(meal_type, self.meal_templates["lunch"])
            
            shopping_list = []
            total_cost = 0.0
            stores_needed = set()
            validation_issues = []
            
            # Find cheapest version of each item with dynamic validation
            for item in base_items:
                if total_cost >= budget:
                    break
                
                # Use dynamic validation for shopping list
                search_result = await self.find_cheapest_product_with_intelligent_suggestions(item)
                
                if search_result["success"] and search_result["products"]:
                    products = search_result["products"]
                    
                    # Filter by dietary restrictions if provided
                    if dietary_restrictions:
                        products = self._filter_by_dietary_restrictions(products, dietary_restrictions)
                    
                    if products:
                        best_product = products[0]  # Already sorted by price
                        quantity_needed = people_count
                        item_cost = best_product['current_price'] * quantity_needed
                        
                        if total_cost + item_cost <= budget:
                            shopping_list.append({
                                **best_product,
                                'quantity': quantity_needed,
                                'total_item_cost': item_cost
                            })
                            total_cost += item_cost
                            stores_needed.add(best_product['store_name'])
                        else:
                            validation_issues.append(f"Budget exceeded for {item}")
                    else:
                        validation_issues.append(f"No products found matching dietary restrictions for {item}")
                else:
                    validation_issues.append(f"No validated products found for {item}")
                    # Add suggestions to validation issues
                    if search_result.get("suggestions"):
                        validation_issues.append(f"Suggestions for {item}: {', '.join(search_result['suggestions'])}")
            
            # Calculate additional statistics
            avg_item_cost = total_cost / len(shopping_list) if shopping_list else 0
            budget_efficiency = (total_cost / budget) * 100 if budget > 0 else 0
            
            result = {
                "shopping_list": shopping_list,
                "total_cost": total_cost,
                "budget_used": total_cost,
                "budget_remaining": budget - total_cost,
                "budget_efficiency": budget_efficiency,
                "average_item_cost": avg_item_cost,
                "stores_needed": list(stores_needed),
                "serves": people_count,
                "meal_type": meal_type,
                "validation_applied": use_semantic_validation,
                "validation_issues": validation_issues,
                "items_found": len(shopping_list),
                "items_requested": len(base_items)
            }
            
            logger.info(f"🛒 Shopping list created: {len(shopping_list)} items, €{total_cost:.2f} cost")
            return result
            
        except Exception as e:
            logger.error(f"Error creating shopping list: {e}")
            return {
                "shopping_list": [],
                "total_cost": 0.0,
                "budget_used": 0.0,
                "budget_remaining": budget,
                "stores_needed": [],
                "serves": people_count,
                "meal_type": meal_type,
                "validation_applied": use_semantic_validation,
                "error": str(e)
            }

    # Helper methods (keep existing ones)
    def _filter_by_dietary_restrictions(self, products: List[Dict], restrictions: List[str]) -> List[Dict]:
        """Filter products by dietary restrictions using LLM understanding"""
        filtered = []
        for product in products:
            category = product.get('ai_main_category', '').lower()
            product_name = product.get('product_name', '').lower()
            diet_compatibility = product.get('ai_diet_compatibility', '').lower()
            
            # Use AI diet compatibility if available
            if diet_compatibility:
                if any(restriction.lower() in diet_compatibility for restriction in restrictions):
                    filtered.append(product)
                    continue
            
            # Fallback to simple filtering logic
            should_include = True
            for restriction in restrictions:
                restriction_lower = restriction.lower()
                
                if restriction_lower == 'vegetarian':
                    if 'meso' in product_name or 'meat' in product_name:
                        should_include = False
                        break
                elif restriction_lower == 'vegan':
                    if any(x in product_name for x in ['meso', 'mleko', 'sir', 'jajca', 'meat', 'milk', 'cheese', 'egg']):
                        should_include = False
                        break
                elif restriction_lower == 'gluten_free':
                    if any(x in product_name for x in ['kruh', 'testenine', 'pšenica', 'bread', 'pasta', 'wheat']):
                        should_include = False
                        break
            
            if should_include:
                filtered.append(product)
        
        return filtered

    async def get_product_insights(self, product_name: str) -> Dict[str, Any]:
        """Get detailed insights about a product using LLM analysis"""
        try:
            # Get product results with validation
            search_result = await self.find_cheapest_product_with_intelligent_suggestions(product_name)
            
            if not search_result["success"] or not search_result["products"]:
                return {
                    "success": False,
                    "message": f"No insights available for '{product_name}'",
                    "suggestions": search_result.get("suggestions", [])
                }
            
            products = search_result["products"][:10]  # Analyze top 10 products
            
            # Generate insights using LLM
            insights_prompt = f"""
            Analyze these grocery products for "{product_name}":

            {json.dumps([{
                "name": p.get("product_name", ""),
                "store": p.get("store_name", ""),
                "price": p.get("current_price", 0),
                "category": p.get("ai_main_category", ""),
                "health_score": p.get("ai_health_score"),
                "has_discount": p.get("has_discount", False),
                "discount_percentage": p.get("discount_percentage")
            } for p in products], indent=2)}

            Provide insights in JSON format:
            {{
                "price_analysis": {{
                    "cheapest_price": 0.0,
                    "most_expensive_price": 0.0,
                    "average_price": 0.0,
                    "best_value_store": "store_name"
                }},
                "health_analysis": {{
                    "average_health_score": 0.0,
                    "healthiest_option": "product_name",
                    "health_recommendation": "text"
                }},
                "savings_opportunities": [
                    {{
                        "store": "store_name",
                        "product": "product_name",
                        "savings": 0.0,
                        "discount_percentage": 0
                    }}
                ],
                "recommendations": [
                    "recommendation 1",
                    "recommendation 2"
                ],
                "summary": "Brief summary of findings"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": insights_prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
            )
            
            insights_text = response.choices[0].message.content.strip()
            
            # Parse insights
            try:
                if "```json" in insights_text:
                    json_text = insights_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_text = insights_text
                    
                insights = json.loads(json_text)
                
                return {
                    "success": True,
                    "product_name": product_name,
                    "insights": insights,
                    "products_analyzed": len(products),
                    "validation_applied": search_result.get("validation_applied", False)
                }
                
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "message": "Failed to parse insights",
                    "raw_response": insights_text
                }
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "success": False,
                "message": f"Failed to generate insights: {str(e)}"
            }

if __name__ == "__main__":
    asyncio.run()