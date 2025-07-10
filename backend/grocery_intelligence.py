#!/usr/bin/env python3
"""
Enhanced Slovenian Grocery Intelligence with Semantic Validation
Prevents wrong products like "MLEČNA REZINA MILKA" when searching for "mleko"
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

# Import the semantic validator
from semantic_search_validation import SemanticSearchValidator, EnhancedProductSearch

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
    semantic_match_score: Optional[float] = None  # New field for validation confidence

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

class SlovenianGroceryMCP:
    """Enhanced grocery intelligence system with semantic validation"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_manager = DatabaseManager(db_config)
        self.semantic_validator = SemanticSearchValidator()
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
            "ai_value_rating": self._safe_int(row.get("ai_value_rating"))
        }
    
    # Enhanced core functionality methods with semantic validation
    async def find_cheapest_product(
        self, 
        product_name: str, 
        location: str = "Ljubljana", 
        store_preference: Optional[str] = None,
        use_semantic_validation: bool = True
    ) -> List[Dict]:
        """Find cheapest product across stores with optional semantic validation"""
        try:
            # Step 1: Get raw database results (more than needed for validation)
            raw_results = await self._get_raw_product_search(product_name, store_preference, limit=100)
            
            if not raw_results:
                logger.info(f"❌ No products found for '{product_name}'")
                return []
            
            logger.info(f"🔍 Found {len(raw_results)} raw results for '{product_name}'")
            
            # Step 2: Apply semantic validation if enabled
            if use_semantic_validation:
                validated_results = await self.semantic_validator.validate_search_results(
                    product_name, raw_results, max_results=50
                )
                
                if not validated_results:
                    logger.warning(f"⚠️ No products passed semantic validation for '{product_name}'")
                    # Return empty with suggestion message
                    return []
                
                logger.info(f"✅ {len(validated_results)} products passed semantic validation")
                return validated_results
            
            # Step 3: Return raw results if validation disabled
            formatted_results = [self._format_product_result(row) for row in raw_results[:50]]
            logger.info(f"📋 Returning {len(formatted_results)} unvalidated results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in find_cheapest_product: {e}")
            return []

    async def _get_raw_product_search(
        self, 
        product_name: str, 
        store_preference: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict]:
        """Get raw search results from database"""
        try:
            # Enhanced query that gets more data for validation
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
            return [self._format_product_result(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error in raw product search: {e}")
            return []

    async def find_cheapest_product_with_suggestions(
        self, 
        product_name: str, 
        location: str = "Ljubljana", 
        store_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced search that provides suggestions when no valid results found"""
        
        # Try validated search first
        validated_results = await self.find_cheapest_product(
            product_name, location, store_preference, use_semantic_validation=True
        )
        
        if validated_results:
            return {
                "success": True,
                "products": validated_results,
                "message": f"Found {len(validated_results)} products matching '{product_name}'",
                "search_term": product_name,
                "validation_applied": True
            }
        
        # If no validated results, try without validation to see what we got
        raw_results = await self.find_cheapest_product(
            product_name, location, store_preference, use_semantic_validation=False
        )
        
        # Generate suggestions
        suggestions = await self._generate_search_suggestions(product_name, raw_results)
        
        return {
            "success": False,
            "products": [],
            "message": f"No matching products found for '{product_name}'",
            "search_term": product_name,
            "suggestions": suggestions,
            "validation_applied": True,
            "raw_results_count": len(raw_results)
        }

    async def _generate_search_suggestions(
        self, 
        product_name: str, 
        raw_results: List[Dict]
    ) -> List[str]:
        """Generate helpful search suggestions"""
        
        suggestions = []
        
        # Analyze what categories the raw results belong to
        if raw_results:
            categories = list(set([r.get('ai_main_category', '') for r in raw_results if r.get('ai_main_category')]))
            
            # Suggest related terms based on categories found
            category_suggestions = {
                "Sladkarije": ["čokolada", "bonboni", "sladice"],
                "Mlečni izdelki": ["mleko", "sir", "jogurt", "maslo"],
                "Pekovski izdelki": ["kruh", "pecivo", "torte"],
                "Sadje": ["jabolka", "banane", "pomaranče"],
                "Zelenjava": ["krompir", "čebula", "paradižnik"],
                "Beverages": ["kava", "čaj", "sok"],
            }
            
            for category in categories:
                if category in category_suggestions:
                    suggestions.extend(category_suggestions[category])
        
        # Add common alternative spellings/terms
        common_alternatives = {
            "mleko": ["milk", "mlečni izdelki"],
            "kruh": ["bread", "pekovski izdelki"],
            "jajca": ["eggs", "jajce"],
            "kava": ["coffee", "kavni napitki"],
            "čaj": ["tea", "čajni napitki"]
        }
        
        product_lower = product_name.lower()
        for term, alternatives in common_alternatives.items():
            if term in product_lower or product_lower in alternatives:
                suggestions.extend([term] + alternatives)
        
        # Remove duplicates and the original term
        suggestions = list(set(suggestions))
        suggestions = [s for s in suggestions if s.lower() != product_name.lower()]
        
        return suggestions[:5]

    async def compare_prices(
        self, 
        product_name: str, 
        stores: Optional[List[str]] = None,
        use_semantic_validation: bool = True
    ) -> Dict[str, List[Dict]]:
        """Compare prices across stores with semantic validation"""
        try:
            # Get results with validation
            if use_semantic_validation:
                all_results = await self.find_cheapest_product(
                    product_name, use_semantic_validation=True
                )
            else:
                query = """
                SELECT store_name, product_name, current_price, regular_price,
                       has_discount, discount_percentage, product_url, 
                       ai_main_category, ai_subcategory
                FROM unified_products_view 
                WHERE product_name LIKE %s 
                AND current_price > 0
                """
                params = [f"%{product_name}%"]
                
                if stores:
                    placeholders = ",".join(["%s"] * len(stores))
                    query += f" AND store_name IN ({placeholders})"
                    params.extend(stores)
                
                query += " ORDER BY store_name, current_price ASC"
                
                results = await self.db_manager.execute_query(query, params)
                all_results = [self._format_product_result(row) for row in results]
            
            # Group by store
            store_results = {}
            for row in all_results:
                store = row['store_name']
                if store not in store_results:
                    store_results[store] = []
                store_results[store].append(row)
            
            logger.info(f"Compared prices across {len(store_results)} stores")
            return store_results
            
        except Exception as e:
            logger.error(f"Error comparing prices: {e}")
            return {}

    # Rest of the methods remain the same but can use validated results
    async def create_budget_shopping_list(
        self, 
        budget: float, 
        meal_type: str, 
        people_count: int = 1, 
        dietary_restrictions: Optional[List[str]] = None,
        use_semantic_validation: bool = True
    ) -> Dict:
        """Create budget-optimized shopping list with validated products"""
        try:
            # Get base items for meal type
            base_items = self.meal_templates.get(meal_type, self.meal_templates["lunch"])
            
            shopping_list = []
            total_cost = 0.0
            stores_needed = set()
            validation_issues = []
            
            # Find cheapest version of each item with validation
            for item in base_items:
                if total_cost >= budget:
                    break
                
                # Use semantic validation for shopping list
                search_result = await self.find_cheapest_product_with_suggestions(item)
                
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
                    if search_result.get("suggestions"):
                        validation_issues.append(f"Try searching for: {', '.join(search_result['suggestions'])}")
            
            result = {
                "shopping_list": shopping_list,
                "total_cost": total_cost,
                "budget_used": total_cost,
                "budget_remaining": budget - total_cost,
                "stores_needed": list(stores_needed),
                "serves": people_count,
                "meal_type": meal_type,
                "validation_applied": use_semantic_validation,
                "validation_issues": validation_issues
            }
            
            logger.info(f"Created shopping list with {len(shopping_list)} items, cost: €{total_cost:.2f}")
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
        """Filter products by dietary restrictions"""
        filtered = []
        for product in products:
            category = product.get('ai_main_category', '').lower()
            
            # Simple filtering logic - can be enhanced
            if 'vegetarian' in restrictions and 'meso' in product['product_name'].lower():
                continue
            if 'vegan' in restrictions and any(x in product['product_name'].lower() for x in ['meso', 'mleko', 'sir', 'jajca']):
                continue
            if 'gluten_free' in restrictions and any(x in product['product_name'].lower() for x in ['kruh', 'testenine', 'pšenica']):
                continue
            
            filtered.append(product)
        
        return filtered

# Enhanced testing
async def test_enhanced_search():
    """Test enhanced search functionality"""
    print("🧪 Testing Enhanced Slovenian Grocery Intelligence with Semantic Validation")
    
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
    
    system = SlovenianGroceryMCP(db_config)
    
    try:
        await system.connect_db()
        
        # Test cases that should show validation improvement
        test_cases = [
            "mleko",  # Should exclude MLEČNA REZINA MILKA
            "kruh",   # Should exclude breadcrumbs
            "jabolka" # Should exclude apple juice
        ]
        
        for test_term in test_cases:
            print(f"\n🔍 Testing search for '{test_term}':")
            
            # Test with validation
            result_with_validation = await system.find_cheapest_product_with_suggestions(test_term)
            
            print(f"✅ With validation: {result_with_validation['success']}")
            if result_with_validation['success']:
                print(f"   Found {len(result_with_validation['products'])} validated products")
                for product in result_with_validation['products'][:3]:
                    print(f"   - {product['product_name']} (€{product['current_price']:.2f})")
            else:
                print(f"   ❌ {result_with_validation['message']}")
                if result_with_validation.get('suggestions'):
                    print(f"   💡 Suggestions: {', '.join(result_with_validation['suggestions'])}")
            
            # Test without validation for comparison
            raw_results = await system.find_cheapest_product(test_term, use_semantic_validation=False)
            print(f"📊 Without validation: {len(raw_results)} raw results")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        system.disconnect_db()

if __name__ == "__main__":
    asyncio.run(test_enhanced_search())