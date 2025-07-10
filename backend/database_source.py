#!/usr/bin/env python3
"""
Enhanced Database Source for Slovenian Grocery Intelligence
Leverages all AI-enhanced columns for intelligent recommendations
"""

import os
import asyncio
import json
from typing import Dict, List, Any, Optional
import pymysql
import logging
from datetime import datetime
from dotenv import load_dotenv
from decimal import Decimal

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDatabaseSource:
    """Enhanced database source with full AI feature support"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        self.available_columns = []
    
    async def connect(self) -> None:
        """Establish database connection and check available columns"""
        try:
            self.connection = pymysql.connect(**self.db_config)
            logger.info("✅ Enhanced database source connected successfully")
            
            # Get available columns
            await self._check_available_columns()
            await self.test_connection()
            
        except Exception as e:
            logger.error(f"❌ Database source connection failed: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
    
    async def _check_available_columns(self) -> None:
        """Check which columns are available in the unified_products_view"""
        try:
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute("DESCRIBE unified_products_view")
            columns = cursor.fetchall()
            self.available_columns = [col['Field'] for col in columns]
            logger.info(f"📊 Available columns: {len(self.available_columns)} AI-enhanced columns detected")
            cursor.close()
        except Exception as e:
            logger.error(f"Error checking columns: {e}")
            self.available_columns = []
    
    
    async def test_connection(self) -> bool:
        """Test database connection and verify AI features"""
        try:
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            
            # Check AI features availability
            ai_features = [col for col in self.available_columns if col.startswith('ai_')]
            logger.info(f"🤖 AI features available: {len(ai_features)} columns")
            
            # Get sample data count
            cursor.execute("SELECT COUNT(*) as count FROM unified_products_view")
            count_result = cursor.fetchone()
            logger.info(f"📈 Products in database: {count_result['count'] if count_result else 0}")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Enhanced database source connection closed")
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """Execute a custom query and return results"""
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute(query, params or [])
            results = cursor.fetchall()
            # Convert Decimal objects to float for JSON serialization
            return [self._convert_decimals(row) for row in results]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    async def get_health_focused_products(self, min_health_score: int = 7, limit: int = 50) -> List[Dict]:
        """Get health-focused product recommendations"""
        try:
            query = """
            SELECT product_name, store_name, current_price, regular_price, has_discount,
                   ai_health_score, ai_nutrition_grade, ai_sugar_content, ai_sodium_level,
                   ai_organic_verified, ai_processing_level, ai_diet_compatibility,
                   ai_allergen_risk, ai_value_rating
            FROM unified_products_view 
            WHERE ai_health_score >= %s 
            AND current_price > 0
            ORDER BY ai_health_score DESC, current_price ASC
            LIMIT %s
            """
            return await self.execute_query(query, [min_health_score, limit])
        except Exception as e:
            logger.error(f"Error getting health-focused products: {e}")
            return []
    
    async def get_diet_compatible_products(self, diet_type: str, limit: int = 50) -> List[Dict]:
        """Get products compatible with specific diets (vegan, vegetarian, keto, etc.)"""
        try:
            query = """
            SELECT product_name, store_name, current_price, ai_main_category,
                   ai_health_score, ai_nutrition_grade, ai_diet_compatibility,
                   ai_allergen_list, ai_allergen_risk, ai_preparation_tips
            FROM unified_products_view 
            WHERE ai_diet_compatibility LIKE %s 
            AND current_price > 0
            ORDER BY ai_health_score DESC, current_price ASC
            LIMIT %s
            """
            return await self.execute_query(query, [f"%{diet_type}%", limit])
        except Exception as e:
            logger.error(f"Error getting diet-compatible products: {e}")
            return []
    
    async def get_meal_planning_suggestions(self, meal_category: str = None, max_prep_complexity: str = "moderate") -> List[Dict]:
        """Get intelligent meal planning suggestions"""
        try:
            query = """
            SELECT product_name, store_name, current_price, ai_meal_category,
                   ai_pairing_suggestions, ai_recipe_compatibility, ai_preparation_complexity,
                   ai_preparation_tips, ai_health_score, ai_stockup_recommendation,
                   ai_optimal_quantity
            FROM unified_products_view 
            WHERE current_price > 0
            """
            params = []
            
            if meal_category:
                query += " AND ai_meal_category = %s"
                params.append(meal_category)
            
            if max_prep_complexity:
                complexity_order = {"simple": 1, "moderate": 2, "complex": 3}
                if max_prep_complexity in complexity_order:
                    query += " AND ai_preparation_complexity IN ('simple'"
                    if complexity_order[max_prep_complexity] >= 2:
                        query += ", 'moderate'"
                    if complexity_order[max_prep_complexity] >= 3:
                        query += ", 'complex'"
                    query += ")"
            
            query += " ORDER BY ai_health_score DESC, current_price ASC LIMIT 30"
            
            return await self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Error getting meal planning suggestions: {e}")
            return []
    
    async def get_smart_shopping_deals(self, min_deal_quality: str = "good") -> List[Dict]:
        """Get intelligent shopping deals based on AI analysis"""
        try:
            query = """
            SELECT product_name, store_name, current_price, regular_price,
                   discount_percentage, ai_deal_quality, ai_value_rating,
                   ai_stockup_recommendation, ai_bulk_discount_worthy,
                   ai_optimal_quantity, ai_replacement_urgency, ai_price_tier
            FROM unified_products_view 
            WHERE has_discount = 1 
            AND current_price > 0
            AND ai_deal_quality IN (%s)
            ORDER BY 
                CASE ai_deal_quality 
                    WHEN 'excellent' THEN 1 
                    WHEN 'good' THEN 2 
                    WHEN 'fair' THEN 3 
                    ELSE 4 
                END,
                discount_percentage DESC
            LIMIT 50
            """
            
            # Build quality filter
            quality_options = []
            if min_deal_quality == "excellent":
                quality_options = ["'excellent'"]
            elif min_deal_quality == "good":
                quality_options = ["'excellent'", "'good'"]
            else:
                quality_options = ["'excellent'", "'good'", "'fair'"]
            
            query = query.replace("(%s)", "(" + ", ".join(quality_options) + ")")
            
            return await self.execute_query(query)
        except Exception as e:
            logger.error(f"Error getting smart shopping deals: {e}")
            return []
    
    async def get_allergen_safe_products(self, avoid_allergens: List[str]) -> List[Dict]:
        """Get products safe for people with specific allergies"""
        try:
            query = """
            SELECT product_name, store_name, current_price, ai_allergen_list,
                   ai_allergen_risk, ai_diet_compatibility, ai_health_score,
                   ai_nutrition_grade, ai_organic_verified
            FROM unified_products_view 
            WHERE ai_allergen_risk = 'low'
            AND current_price > 0
            """
            
            # Add allergen exclusions
            for allergen in avoid_allergens:
                query += f" AND (ai_allergen_list NOT LIKE '%{allergen}%' OR ai_allergen_list IS NULL)"
            
            query += " ORDER BY ai_health_score DESC, current_price ASC LIMIT 50"
            
            return await self.execute_query(query)
        except Exception as e:
            logger.error(f"Error getting allergen-safe products: {e}")
            return []
    
    async def get_recipe_ingredients(self, recipe_type: str, budget: float = None) -> List[Dict]:
        """Get ingredients for specific recipe types with budget consideration"""
        try:
            query = """
            SELECT product_name, store_name, current_price, ai_main_category,
                   ai_recipe_compatibility, ai_pairing_suggestions, ai_preparation_tips,
                   ai_optimal_quantity, ai_storage_requirements, ai_shelf_life_estimate
            FROM unified_products_view 
            WHERE ai_recipe_compatibility LIKE %s 
            AND current_price > 0
            """
            params = [f"%{recipe_type}%"]
            
            if budget:
                query += " AND current_price <= %s"
                params.append(budget)
            
            query += " ORDER BY ai_health_score DESC, current_price ASC LIMIT 30"
            
            return await self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Error getting recipe ingredients: {e}")
            return []
    
    async def get_seasonal_recommendations(self, season: str = None) -> List[Dict]:
        """Get seasonal product recommendations"""
        try:
            query = """
            SELECT product_name, store_name, current_price, ai_seasonal_availability,
                   ai_freshness_indicator, ai_health_score, ai_environmental_score,
                   ai_main_category, ai_pairing_suggestions
            FROM unified_products_view 
            WHERE current_price > 0
            """
            params = []
            
            if season:
                query += " AND ai_seasonal_availability LIKE %s"
                params.append(f"%{season}%")
            
            query += """
            ORDER BY ai_freshness_indicator DESC, ai_environmental_score DESC, 
                     current_price ASC 
            LIMIT 40
            """
            
            return await self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Error getting seasonal recommendations: {e}")
            return []
    
    async def get_storage_and_freshness_tips(self, product_name: str) -> List[Dict]:
        """Get detailed storage and freshness information for products"""
        try:
            query = """
            SELECT product_name, store_name, ai_storage_requirements, ai_storage_tips,
                   ai_shelf_life_estimate, ai_freshness_indicator, ai_preparation_tips,
                   ai_usage_suggestions, ai_alternative_uses
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND current_price > 0
            ORDER BY ai_shelf_life_estimate DESC
            LIMIT 10
            """
            return await self.execute_query(query, [f"%{product_name}%"])
        except Exception as e:
            logger.error(f"Error getting storage tips: {e}")
            return []
    
    async def get_environmental_impact_analysis(self, min_env_score: int = 6) -> List[Dict]:
        """Get products with good environmental impact scores"""
        try:
            query = """
            SELECT product_name, store_name, current_price, ai_environmental_score,
                   ai_organic_verified, ai_processing_level, ai_main_category,
                   ai_health_score, ai_value_rating
            FROM unified_products_view 
            WHERE ai_environmental_score >= %s 
            AND current_price > 0
            ORDER BY ai_environmental_score DESC, ai_organic_verified DESC, current_price ASC
            LIMIT 40
            """
            return await self.execute_query(query, [min_env_score])
        except Exception as e:
            logger.error(f"Error getting environmental analysis: {e}")
            return []
    
    async def get_value_analysis(self, price_tier: str = None) -> List[Dict]:
        """Get detailed value analysis for products"""
        try:
            query = """
            SELECT product_name, store_name, current_price, regular_price,
                   ai_value_rating, ai_price_tier, ai_quality_tier, ai_deal_quality,
                   ai_health_score, has_discount, discount_percentage
            FROM unified_products_view 
            WHERE current_price > 0
            """
            params = []
            
            if price_tier:
                query += " AND ai_price_tier = %s"
                params.append(price_tier)
            
            query += """
            ORDER BY 
                CASE ai_value_rating 
                    WHEN 'excellent' THEN 1 
                    WHEN 'good' THEN 2 
                    WHEN 'fair' THEN 3 
                    ELSE 4 
                END,
                current_price ASC
            LIMIT 50
            """
            
            return await self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Error getting value analysis: {e}")
            return []
    
    def _convert_decimals(self, row: Dict) -> Dict:
        """Convert Decimal objects to float for JSON serialization"""
        converted = {}
        for key, value in row.items():
            if isinstance(value, Decimal):
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted

    async def get_comprehensive_product_analysis(self, product_name: str) -> List[Dict]:
        """Get comprehensive AI analysis for a specific product"""
        try:
            query = """
            SELECT product_name, store_name, current_price, regular_price, has_discount,
                   ai_health_score, ai_nutrition_grade, ai_environmental_score,
                   ai_value_rating, ai_quality_tier, ai_main_category, ai_subcategory,
                   ai_diet_compatibility, ai_allergen_list, ai_allergen_risk,
                   ai_pairing_suggestions, ai_preparation_tips, ai_usage_suggestions,
                   ai_storage_requirements, ai_shelf_life_estimate, ai_product_summary,
                   ai_stockup_recommendation, ai_optimal_quantity, ai_target_demographic,
                   ai_key_selling_points
            FROM unified_products_view 
            WHERE product_name LIKE %s 
            AND current_price > 0
            ORDER BY ai_health_score DESC, current_price ASC
            """
            return await self.execute_query(query, [f"%{product_name}%"])
        except Exception as e:
            logger.error(f"Error getting comprehensive analysis: {e}")
            return []

# Database configuration helper
def get_database_config() -> Dict[str, Any]:
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'ai_food'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', 'root'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'charset': 'utf8mb4',
        'autocommit': True
    }

# Test the enhanced database features
async def test_enhanced_features():
    """Test enhanced database features"""
    print("🤖 Testing enhanced AI database features...")
    
    db_config = get_database_config()
    db_source = EnhancedDatabaseSource(db_config)
    
    try:
        await db_source.connect()
        
        # Test health-focused products
        print("\n🏥 Health-focused products:")
        health_products = await db_source.get_health_focused_products(min_health_score=7, limit=5)
        for product in health_products[:3]:
            print(f"   - {product['product_name']}: Health {product['ai_health_score']}/10, Grade {product['ai_nutrition_grade']}")
        
        # Test diet compatibility
        print("\n🌱 Vegan products:")
        vegan_products = await db_source.get_diet_compatible_products("vegan", limit=3)
        for product in vegan_products:
            print(f"   - {product['product_name']}: {product['ai_diet_compatibility']}")
        
        # Test smart deals
        print("\n💰 Smart shopping deals:")
        deals = await db_source.get_smart_shopping_deals("good")
        for deal in deals[:3]:
            print(f"   - {deal['product_name']}: {deal['discount_percentage']}% off, Deal Quality: {deal['ai_deal_quality']}")
        
        # Test meal planning
        print("\n🍽️ Breakfast meal suggestions:")
        breakfast = await db_source.get_meal_planning_suggestions("breakfast", "simple")
        for meal in breakfast[:3]:
            print(f"   - {meal['product_name']}: Prep: {meal['ai_preparation_complexity']}")
        
        print("\n✅ Enhanced database features test completed successfully!")
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        
    finally:
        db_source.disconnect()

if __name__ == "__main__":
    # Run enhanced features test
    asyncio.run(test_enhanced_features())