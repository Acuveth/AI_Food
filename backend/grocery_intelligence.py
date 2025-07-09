#!/usr/bin/env python3
"""
Slovenian Grocery Intelligence MCP Server (Enhanced Version)
AI-powered grocery shopping and meal planning system for Slovenian stores
Uses PyMySQL for better MariaDB compatibility with improved error handling
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@dataclass
class ShoppingListItem:
    """Represents an item in a shopping list"""
    product: ProductResult
    quantity: int
    total_cost: float

@dataclass
class ShoppingList:
    """Represents a complete shopping list"""
    items: List[ShoppingListItem]
    total_cost: float
    stores_needed: List[str]
    budget_used: float
    budget_remaining: float
    meal_type: str
    serves: int

@dataclass
class Promotion:
    """Represents a promotional offer"""
    product: ProductResult
    savings: float
    valid_until: Optional[datetime] = None
    promotion_type: str = "discount"

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
    """Main grocery intelligence system"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_manager = DatabaseManager(db_config)
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
    
    # Core functionality methods
    async def find_cheapest_product(
        self, 
        product_name: str, 
        location: str = "Ljubljana", 
        store_preference: Optional[str] = None
    ) -> List[Dict]:
        """Find cheapest product across stores"""
        try:
            query = """
            SELECT store_name, product_name, current_price, regular_price, 
                   has_discount, discount_percentage, product_url, 
                   ai_main_category, ai_subcategory, ai_confidence,
                   ai_health_score, ai_nutrition_grade, ai_value_rating
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND current_price > 0
            """
            params = [f"%{product_name}%"]
            
            if store_preference:
                query += " AND store_name = %s"
                params.append(store_preference)
            
            query += " ORDER BY current_price ASC LIMIT 50"
            
            results = await self.db_manager.execute_query(query, params)
            
            # Format results
            formatted_results = [self._format_product_result(row) for row in results]
            
            logger.info(f"Found {len(formatted_results)} products for '{product_name}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error finding cheapest product: {e}")
            return []
    
    async def compare_prices(
        self, 
        product_name: str, 
        stores: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """Compare prices across stores"""
        try:
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
            
            # Group by store
            store_results = {}
            for row in results:
                store = row['store_name']
                if store not in store_results:
                    store_results[store] = []
                store_results[store].append(self._format_product_result(row))
            
            logger.info(f"Compared prices across {len(store_results)} stores")
            return store_results
            
        except Exception as e:
            logger.error(f"Error comparing prices: {e}")
            return {}
    
    async def create_budget_shopping_list(
        self, 
        budget: float, 
        meal_type: str, 
        people_count: int = 1, 
        dietary_restrictions: Optional[List[str]] = None
    ) -> Dict:
        """Create budget-optimized shopping list"""
        try:
            # Get base items for meal type
            base_items = self.meal_templates.get(meal_type, self.meal_templates["lunch"])
            
            shopping_list = []
            total_cost = 0.0
            stores_needed = set()
            
            # Find cheapest version of each item
            for item in base_items:
                if total_cost >= budget:
                    break
                
                products = await self.find_cheapest_product(item, "Ljubljana")
                
                if products:
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
            
            result = {
                "shopping_list": shopping_list,
                "total_cost": total_cost,
                "budget_used": total_cost,
                "budget_remaining": budget - total_cost,
                "stores_needed": list(stores_needed),
                "serves": people_count,
                "meal_type": meal_type
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
                "meal_type": meal_type
            }
    
    async def get_current_promotions(
        self, 
        store: Optional[str] = None, 
        category: Optional[str] = None, 
        min_discount: int = 10
    ) -> List[Dict]:
        """Get current promotions and discounts"""
        try:
            query = """
            SELECT store_name, product_name, current_price, regular_price,
                   discount_percentage, product_url, ai_main_category, ai_subcategory
            FROM unified_products_view 
            WHERE has_discount = 1 
            AND discount_percentage >= %s
            AND current_price > 0
            """
            params = [min_discount]
            
            if store:
                query += " AND store_name = %s"
                params.append(store)
            
            if category:
                query += " AND ai_main_category LIKE %s"
                params.append(f"%{category}%")
            
            query += " ORDER BY discount_percentage DESC, current_price ASC LIMIT 100"
            
            results = await self.db_manager.execute_query(query, params)
            
            promotions = []
            for row in results:
                promotion = self._format_product_result(row)
                regular_price = self._safe_float(row.get('regular_price', 0)) or 0
                current_price = self._safe_float(row.get('current_price', 0)) or 0
                promotion['savings'] = max(0, regular_price - current_price)
                promotions.append(promotion)
            
            logger.info(f"Found {len(promotions)} promotions with {min_discount}%+ discount")
            return promotions
            
        except Exception as e:
            logger.error(f"Error getting promotions: {e}")
            return []
    
    async def get_store_availability(self, product_name: str) -> Dict[str, bool]:
        """Check store availability for a product"""
        try:
            query = """
            SELECT DISTINCT store_name
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND current_price > 0
            """
            
            results = await self.db_manager.execute_query(query, [f"%{product_name}%"])
            available_stores = [row['store_name'] for row in results]
            
            # Check all known stores
            availability = {}
            for store in self.stores:
                availability[store] = store in available_stores
            
            logger.info(f"Checked availability for '{product_name}' across {len(self.stores)} stores")
            return availability
            
        except Exception as e:
            logger.error(f"Error checking store availability: {e}")
            return {store: False for store in self.stores}
    
    async def get_ai_insights(self, product_name: str) -> Dict:
        """Get AI insights for a product"""
        try:
            query = """
            SELECT store_name, product_name, current_price, ai_main_category, ai_subcategory,
                   ai_confidence, ai_health_score, ai_nutrition_grade, ai_diet_compatibility,
                   ai_environmental_score, ai_value_rating, ai_product_summary
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND current_price > 0
            ORDER BY current_price ASC
            LIMIT 20
            """
            
            results = await self.db_manager.execute_query(query, [f"%{product_name}%"])
            
            if not results:
                return {"message": f"No products found for '{product_name}'"}
            
            # Process results
            prices = [self._safe_float(r['current_price']) for r in results if self._safe_float(r['current_price'])]
            categories = list(set([r['ai_main_category'] for r in results if r['ai_main_category']]))
            health_scores = [self._safe_float(r['ai_health_score']) for r in results if self._safe_float(r['ai_health_score'])]
            nutrition_grades = list(set([r['ai_nutrition_grade'] for r in results if r['ai_nutrition_grade']]))
            
            insights = {
                "product_count": len(results),
                "price_range": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                    "avg": sum(prices) / len(prices) if prices else 0
                },
                "categories": categories,
                "health_scores": health_scores,
                "avg_health_score": sum(health_scores) / len(health_scores) if health_scores else None,
                "nutrition_grades": nutrition_grades,
                "products": [self._format_product_result(row) for row in results]
            }
            
            logger.info(f"Generated insights for '{product_name}': {len(results)} products analyzed")
            return insights
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return {"message": f"Error analyzing '{product_name}': {str(e)}"}
    
    async def suggest_meal_from_promotions(
        self, 
        budget: float, 
        meal_type: str, 
        people_count: int = 1
    ) -> Dict:
        """Suggest meal based on current promotions"""
        try:
            # Get high-discount promotions
            promotions = await self.get_current_promotions(min_discount=20)
            
            if not promotions:
                return {
                    "message": "No suitable promotions found",
                    "meal_suggestion": None
                }
            
            # Group by category
            categories = {}
            for promo in promotions:
                category = promo.get('ai_main_category', 'Other')
                if category not in categories:
                    categories[category] = []
                categories[category].append(promo)
            
            # Build meal from different categories
            meal_items = []
            total_cost = 0.0
            total_savings = 0.0
            
            # Try to get items from different categories
            for category, items in list(categories.items())[:4]:
                if total_cost >= budget:
                    break
                
                # Sort by savings
                items.sort(key=lambda x: x.get('savings', 0), reverse=True)
                
                for item in items:
                    if total_cost + item['current_price'] <= budget:
                        meal_items.append(item)
                        total_cost += item['current_price']
                        total_savings += item.get('savings', 0)
                        break
            
            # Generate recipe
            recipe = self._generate_simple_recipe(meal_items, meal_type, people_count)
            
            result = {
                "meal_suggestion": {
                    "name": f"Promotional {meal_type.title()}",
                    "ingredients": meal_items,
                    "total_cost": total_cost,
                    "total_savings": total_savings,
                    "serves": people_count,
                    "recipe": recipe
                },
                "budget_used": total_cost,
                "budget_remaining": budget - total_cost
            }
            
            logger.info(f"Created promotional meal: {len(meal_items)} items, €{total_cost:.2f}, saves €{total_savings:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error suggesting meal from promotions: {e}")
            return {
                "message": f"Error creating meal suggestion: {str(e)}",
                "meal_suggestion": None
            }
    
    # Helper methods
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
    
    def _generate_simple_recipe(self, ingredients: List[Dict], meal_type: str, serves: int) -> str:
        """Generate simple recipe from ingredients"""
        ingredient_names = [item['product_name'] for item in ingredients]
        
        recipes = {
            "breakfast": f"Pripravite zajtrk za {serves} oseb z naslednjimi sestavinami: {', '.join(ingredient_names)}. Kombinirajte sestavine po okusu za zdrav in hranljiv zajtrk.",
            "lunch": f"Pripravite kosilo za {serves} oseb. Uporabite {', '.join(ingredient_names)}. Skuhajte osnovne sestavine in dodajte začimbe po okusu.",
            "dinner": f"Pripravite večerjo za {serves} oseb z {', '.join(ingredient_names)}. Kombinirajte sestavine v okusno in nasitno jed.",
            "snack": f"Pripravite prigrizek z {', '.join(ingredient_names)}. Enostavno kombinirajte sestavine."
        }
        
        return recipes.get(meal_type, recipes["lunch"])
    
    async def get_weekly_meal_plan(self, budget: float, people_count: int = 1) -> Dict:
        """Generate weekly meal plan within budget"""
        try:
            weekly_budget = budget / 7  # Daily budget
            meal_plan = {}
            total_cost = 0.0
            
            days = ['ponedeljek', 'torek', 'sreda', 'četrtek', 'petek', 'sobota', 'nedelja']
            meal_types = ['breakfast', 'lunch', 'dinner']
            
            for day in days:
                meal_plan[day] = {}
                day_cost = 0.0
                
                for meal_type in meal_types:
                    if day_cost >= weekly_budget:
                        break
                    
                    remaining_budget = weekly_budget - day_cost
                    
                    shopping_list = await self.create_budget_shopping_list(
                        remaining_budget, 
                        meal_type, 
                        people_count
                    )
                    
                    if shopping_list['shopping_list']:
                        meal_plan[day][meal_type] = shopping_list
                        day_cost += shopping_list['total_cost']
                
                total_cost += day_cost
            
            return {
                "weekly_plan": meal_plan,
                "total_cost": total_cost,
                "budget_used": total_cost,
                "budget_remaining": budget - total_cost,
                "serves": people_count
            }
            
        except Exception as e:
            logger.error(f"Error creating weekly meal plan: {e}")
            return {
                "weekly_plan": {},
                "total_cost": 0.0,
                "budget_used": 0.0,
                "budget_remaining": budget,
                "serves": people_count
            }
    
    async def get_nutrition_analysis(self, product_name: str) -> Dict:
        """Get nutrition analysis for a product"""
        try:
            query = """
            SELECT ai_health_score, ai_nutrition_grade, ai_diet_compatibility,
                   ai_environmental_score, ai_product_summary
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND ai_health_score IS NOT NULL
            LIMIT 10
            """
            
            results = await self.db_manager.execute_query(query, [f"%{product_name}%"])
            
            if not results:
                return {"message": f"No nutrition data found for '{product_name}'"}
            
            # Calculate averages
            health_scores = [self._safe_float(r['ai_health_score']) for r in results if self._safe_float(r['ai_health_score'])]
            env_scores = [self._safe_float(r['ai_environmental_score']) for r in results if self._safe_float(r['ai_environmental_score'])]
            
            analysis = {
                "product_name": product_name,
                "samples_analyzed": len(results),
                "average_health_score": sum(health_scores) / len(health_scores) if health_scores else None,
                "average_environmental_score": sum(env_scores) / len(env_scores) if env_scores else None,
                "nutrition_grades": list(set([r['ai_nutrition_grade'] for r in results if r['ai_nutrition_grade']])),
                "diet_compatibility": [r['ai_diet_compatibility'] for r in results if r['ai_diet_compatibility']],
                "health_recommendation": self._get_health_recommendation(health_scores)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting nutrition analysis: {e}")
            return {"message": f"Error analyzing nutrition for '{product_name}': {str(e)}"}
    
    def _get_health_recommendation(self, health_scores: List[float]) -> str:
        """Generate health recommendation based on scores"""
        if not health_scores:
            return "Ni podatkov o zdravju"
        
        avg_score = sum(health_scores) / len(health_scores)
        
        if avg_score >= 8:
            return "Odličen izbor za zdravo prehrano"
        elif avg_score >= 6:
            return "Dober izbor z zmerno zdravstveno vrednostjo"
        elif avg_score >= 4:
            return "Povprečen izbor, razmislite o alternativah"
        else:
            return "Ni priporočeno za redno uživanje"

# Context manager for database operations
@asynccontextmanager
async def grocery_system(db_config: Dict[str, Any]):
    """Context manager for grocery system operations"""
    system = SlovenianGroceryMCP(db_config)
    try:
        await system.connect_db()
        yield system
    finally:
        system.disconnect_db()

# CLI interface for testing
class GroceryIntelligenceCLI:
    """Command-line interface for testing the grocery system"""
    
    def __init__(self, mcp: SlovenianGroceryMCP):
        self.mcp = mcp
    
    async def run(self):
        """Run interactive CLI"""
        print("🛒 Slovenian Grocery Intelligence System v2.0")
        print("=" * 50)
        print("AI-powered grocery shopping for Slovenia!")
        print("Enhanced with better error handling and features.")
        
        while True:
            print("\n🎯 Available Options:")
            print("1. 🔍 Find cheapest product")
            print("2. 💰 Create budget shopping list")
            print("3. 🎁 Check current promotions")
            print("4. 🍽️ Suggest meal from promotions")
            print("5. 🏪 Check store availability")
            print("6. ⚖️ Compare prices across stores")
            print("7. 🧠 Get AI insights for product")
            print("8. 📊 Get nutrition analysis")
            print("9. 📅 Create weekly meal plan")
            print("10. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-10): ").strip()
            
            try:
                if choice == "1":
                    await self._find_cheapest_product()
                elif choice == "2":
                    await self._create_budget_shopping_list()
                elif choice == "3":
                    await self._check_current_promotions()
                elif choice == "4":
                    await self._suggest_meal_from_promotions()
                elif choice == "5":
                    await self._check_store_availability()
                elif choice == "6":
                    await self._compare_prices()
                elif choice == "7":
                    await self._get_ai_insights()
                elif choice == "8":
                    await self._get_nutrition_analysis()
                elif choice == "9":
                    await self._create_weekly_meal_plan()
                elif choice == "10":
                    print("👋 Hvala za uporabo Slovenian Grocery Intelligence!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")
    
    async def _find_cheapest_product(self):
        """Find cheapest product interface"""
        product = input("🔍 Product name: ").strip()
        if not product:
            print("❌ Please enter a product name.")
            return
        
        print(f"🔎 Searching for cheapest '{product}'...")
        results = await self.mcp.find_cheapest_product(product)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results[:10], 1):
                price_info = f"€{result['current_price']:.2f}"
                if result['has_discount']:
                    price_info += f" (🔥 {result['discount_percentage']}% OFF)"
                print(f"  {i}. {result['product_name']}")
                print(f"     🏪 {result['store_name'].upper()} - {price_info}")
        else:
            print(f"❌ No results found for '{product}'.")
    
    async def _create_budget_shopping_list(self):
        """Create budget shopping list interface"""
        try:
            budget = float(input("💰 Budget (EUR): ").strip())
            meal_type = input("🍽️ Meal type (breakfast/lunch/dinner/snack): ").strip().lower()
            people = int(input("👥 Number of people: ").strip())
            
            if meal_type not in ['breakfast', 'lunch', 'dinner', 'snack']:
                meal_type = 'lunch'
            
            print(f"🛒 Creating shopping list...")
            result = await self.mcp.create_budget_shopping_list(budget, meal_type, people)
            
            if result['shopping_list']:
                print(f"✅ Shopping list created (€{result['total_cost']:.2f}):")
                for item in result['shopping_list']:
                    print(f"  • {item['quantity']}x {item['product_name']}")
                    print(f"    🏪 {item['store_name'].upper()} - €{item['total_item_cost']:.2f}")
                
                print(f"\n💰 Budget: €{result['budget_used']:.2f} used, €{result['budget_remaining']:.2f} remaining")
                print(f"🏪 Stores: {', '.join(result['stores_needed'])}")
            else:
                print("❌ Could not create shopping list within budget.")
                
        except ValueError:
            print("❌ Please enter valid numbers.")
    
    async def _check_current_promotions(self):
        """Check current promotions interface"""
        min_discount = input("🎁 Minimum discount % (default 15): ").strip() or "15"
        try:
            promotions = await self.mcp.get_current_promotions(min_discount=int(min_discount))
            
            if promotions:
                print(f"🎯 Found {len(promotions)} promotions:")
                for i, promo in enumerate(promotions[:15], 1):
                    savings = promo.get('savings', 0)
                    print(f"  {i}. {promo['product_name']}")
                    print(f"     🏪 {promo['store_name'].upper()} - €{promo['current_price']:.2f}")
                    print(f"     🔥 {promo['discount_percentage']}% OFF (Save €{savings:.2f})")
            else:
                print(f"❌ No promotions found with {min_discount}%+ discount.")
        except ValueError:
            print("❌ Please enter a valid discount percentage.")
    
    async def _suggest_meal_from_promotions(self):
        """Suggest meal from promotions interface"""
        try:
            budget = float(input("💰 Budget (EUR): ").strip())
            meal_type = input("🍽️ Meal type (breakfast/lunch/dinner): ").strip().lower()
            people = int(input("👥 Number of people: ").strip())
            
            if meal_type not in ['breakfast', 'lunch', 'dinner']:
                meal_type = 'lunch'
            
            print(f"🎁 Finding promotional meal...")
            result = await self.mcp.suggest_meal_from_promotions(budget, meal_type, people)
            
            if result.get('meal_suggestion'):
                meal = result['meal_suggestion']
                print(f"✅ Suggested meal: {meal['name']}")
                print(f"💰 Cost: €{meal['total_cost']:.2f}, Savings: €{meal['total_savings']:.2f}")
                print(f"🛒 Ingredients:")
                for ingredient in meal['ingredients']:
                    print(f"  • {ingredient['product_name']}")
                    print(f"    🏪 {ingredient['store_name'].upper()} - €{ingredient['current_price']:.2f}")
                print(f"📝 Recipe: {meal['recipe']}")
            else:
                print("❌ Could not create meal suggestion from current promotions.")
                
        except ValueError:
            print("❌ Please enter valid numbers.")
    
    async def _check_store_availability(self):
        """Check store availability interface"""
        product = input("🏪 Product name: ").strip()
        if not product:
            print("❌ Please enter a product name.")
            return
        
        print(f"🔎 Checking availability for '{product}'...")
        availability = await self.mcp.get_store_availability(product)
        
        print(f"🏪 Store availability:")
        for store, available in availability.items():
            status = "✅ Available" if available else "❌ Not found"
            print(f"  {store.upper()}: {status}")
    
    async def _compare_prices(self):
        """Compare prices interface"""
        product = input("⚖️ Product name: ").strip()
        if not product:
            print("❌ Please enter a product name.")
            return
        
        print(f"⚖️ Comparing prices for '{product}'...")
        comparison = await self.mcp.compare_prices(product)
        
        if comparison:
            print(f"📊 Price comparison:")
            for store, products in comparison.items():
                if products:
                    cheapest = products[0]
                    print(f"  🏪 {store.upper()}: €{cheapest['current_price']:.2f}")
                    if cheapest['has_discount']:
                        print(f"      🔥 {cheapest['discount_percentage']}% OFF")
                else:
                    print(f"  🏪 {store.upper()}: Not available")
        else:
            print(f"❌ No products found for '{product}'.")
    
    async def _get_ai_insights(self):
        """Get AI insights interface"""
        product = input("🧠 Product name: ").strip()
        if not product:
            print("❌ Please enter a product name.")
            return
        
        print(f"🧠 Getting AI insights for '{product}'...")
        insights = await self.mcp.get_ai_insights(product)
        
        if insights.get("message"):
            print(f"❌ {insights['message']}")
            return
        
        print(f"🧠 AI Insights:")
        print(f"  📊 Products analyzed: {insights['product_count']}")
        print(f"  💰 Price range: €{insights['price_range']['min']:.2f} - €{insights['price_range']['max']:.2f}")
        print(f"  📈 Average price: €{insights['price_range']['avg']:.2f}")
        
        if insights['categories']:
            print(f"  📂 Categories: {', '.join(insights['categories'])}")
        
        if insights.get('avg_health_score'):
            print(f"  🏥 Average health score: {insights['avg_health_score']:.1f}/10")
    
    async def _get_nutrition_analysis(self):
        """Get nutrition analysis interface"""
        product = input("📊 Product name: ").strip()
        if not product:
            print("❌ Please enter a product name.")
            return
        
        print(f"📊 Getting nutrition analysis for '{product}'...")
        analysis = await self.mcp.get_nutrition_analysis(product)
        
        if analysis.get("message"):
            print(f"❌ {analysis['message']}")
            return
        
        print(f"📊 Nutrition Analysis:")
        print(f"  🔬 Samples analyzed: {analysis['samples_analyzed']}")
        if analysis.get('average_health_score'):
            print(f"  🏥 Average health score: {analysis['average_health_score']:.1f}/10")
        if analysis.get('average_environmental_score'):
            print(f"  🌍 Environmental score: {analysis['average_environmental_score']:.1f}/10")
        if analysis.get('health_recommendation'):
            print(f"  💡 Recommendation: {analysis['health_recommendation']}")
    
    async def _create_weekly_meal_plan(self):
        """Create weekly meal plan interface"""
        try:
            budget = float(input("💰 Weekly budget (EUR): ").strip())
            people = int(input("👥 Number of people: ").strip())
            
            print(f"📅 Creating weekly meal plan...")
            result = await self.mcp.get_weekly_meal_plan(budget, people)
            
            if result['weekly_plan']:
                print(f"✅ Weekly meal plan created (€{result['total_cost']:.2f}):")
                for day, meals in result['weekly_plan'].items():
                    print(f"  📅 {day.title()}:")
                    for meal_type, meal_data in meals.items():
                        if meal_data.get('shopping_list'):
                            cost = meal_data['total_cost']
                            items = len(meal_data['shopping_list'])
                            print(f"    🍽️ {meal_type.title()}: {items} items, €{cost:.2f}")
                
                print(f"\n💰 Weekly budget: €{result['budget_used']:.2f} used, €{result['budget_remaining']:.2f} remaining")
            else:
                print("❌ Could not create weekly meal plan within budget.")
                
        except ValueError:
            print("❌ Please enter valid numbers.")

# Main function for testing
async def main():
    """Main function for testing the grocery system"""
    print("🛒 Starting Slovenian Grocery Intelligence System...")
    
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
    
    # Use context manager for proper resource management
    async with grocery_system(db_config) as mcp:
        # Test basic functionality
        print("🧪 Testing basic functionality...")
        test_results = await mcp.find_cheapest_product("mleko")
        print(f"✅ System working! Found {len(test_results)} products")
        
        # Run interactive CLI
        cli = GroceryIntelligenceCLI(mcp)
        await cli.run()

if __name__ == "__main__":
    asyncio.run(main())