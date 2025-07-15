#!/usr/bin/env python3
"""
Database Lookup and Interpretation Module - SLOVENIAN SUPPORT
Handles all database operations and contains LLM definitions for each table/row
Enhanced with Slovenian search terms and prompts
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import pymysql
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatabaseHandler:
    """
    Centralized database handler with LLM-powered data interpretation
    Enhanced for Slovenian language support
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        
        # LLM definitions for database understanding
        self.table_definitions = {
            "unified_products_view": {
                "description": "Main view containing all grocery products from Slovenian stores",
                "columns": {
                    "product_name": "Name of the grocery product (usually in Slovenian)",
                    "store_name": "Store where product is sold (dm, lidl, mercator, spar, tus)",
                    "current_price": "Current selling price in EUR",
                    "regular_price": "Original price before discount in EUR",
                    "has_discount": "Boolean indicating if product is on sale",
                    "discount_percentage": "Percentage discount if on sale",
                    "ai_main_category": "AI-classified main category (e.g., dairy, meat, vegetables)",
                    "ai_subcategory": "AI-classified subcategory for more specific grouping",
                    "ai_health_score": "AI-assigned health score from 0-10",
                    "ai_nutrition_grade": "AI-assigned nutrition grade (A-E)",
                    "ai_diet_compatibility": "AI-determined diet compatibility (vegan, vegetarian, etc.)",
                    "ai_allergen_list": "AI-detected allergens in the product",
                    "ai_product_summary": "AI-generated product description"
                },
                "usage": "Primary table for all product searches, price comparisons, and meal ingredient lookups"
            }
        }
        
        # Enhanced LLM prompts for Slovenian language support
        self.llm_prompts = {
            "ingredient_search": """
            You are searching for grocery ingredients in a Slovenian grocery database.
            The user is looking for: "{search_term}"
            
            Generate smart search variations that would find this ingredient in Slovenian stores.
            
            IMPORTANT GUIDELINES:
            1. Prioritize Slovenian terms (products are primarily named in Slovenian)
            2. Include common brand names found in Slovenia
            3. Consider regional variations and spelling
            4. Include both singular and plural forms
            5. Think about how products are actually labeled in Slovenian stores
            
            Generate variations for:
            1. Direct Slovenian translations and common terms
            2. Brand names and variations commonly found in Slovenia
            3. Related product categories
            4. Alternative names and regional terms
            
            Examples:
            - "milk" → ["mleko", "polnomastno mleko", "delno posneto mleko", "Ljubljanske mlekarne", "kravje mleko"]
            - "bread" → ["kruh", "bel kruh", "črn kruh", "polnozrnati kruh", "toast", "žemlja"]
            - "chicken" → ["piščanec", "piščančje meso", "piščančji file", "perutnina", "piščančje prsi"]
            
            IMPORTANT: Respond with valid JSON only. No additional text or formatting.
            
            {{
                "search_terms": ["term1", "term2", "term3", "term4", "term5"],
                "category_hints": ["category1", "category2"]
            }}
            """,
            
            "promotion_analysis": """
            Analiziraj te promocijske izdelke in jih kategoriziraj za uporabnika.
            Izdelki: {products}
            
            Podaj vpoglede o:
            1. Najboljše ponudbe po kategorijah
            2. Trgovine z največ promocij
            3. Sezonski vzorci, če obstajajo
            4. Priporočila
            
            Odgovori v slovenščini in podaj strukturno analizo.
            """,
            
            "price_comparison": """
            Primerjaj te cene za isti izdelek v različnih trgovinah:
            {price_data}
            
            Podaj:
            1. Najcenejšo možnost
            2. Razlike v cenah
            3. Priporočila trgovin
            4. Analizo vrednosti
            
            Odgovori v slovenščini z strukturirano primerjavo.
            """
        }
        
        # Slovenian product category mappings
        self.slovenian_categories = {
            "mleko": ["mleko", "mlečni izdelki", "kravje mleko", "kozje mleko"],
            "kruh": ["kruh", "pekovski izdelki", "pecivo", "toast", "žemlja"],
            "sir": ["sir", "mlečni izdelki", "trdi sir", "mehki sir", "cottage sir"],
            "meso": ["meso", "mesni izdelki", "piščanec", "goveje", "svinjina"],
            "riba": ["riba", "morski sadeži", "losos", "tuna", "postrv"],
            "zelenjava": ["zelenjava", "sveža zelenjava", "krompir", "čebula", "paradižnik"],
            "sadje": ["sadje", "sveže sadje", "jabolka", "banane", "pomaranče"],
            "testenine": ["testenine", "riž", "žitarice", "špageti", "makaroni"],
            "pijače": ["pijače", "sokovi", "mineralna voda", "gazirana pijača"]
        }
    
    async def connect(self):
        """Establish database connection"""
        try:
            self.connection = pymysql.connect(**self.db_config)
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Database disconnected")
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """Execute database query and return results"""
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute(query, params or [])
            results = cursor.fetchall()
            # Convert any Decimal objects to float for JSON serialization
            return [self._convert_decimals(row) for row in results]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    def _convert_decimals(self, row: Dict) -> Dict:
        """Convert Decimal objects to float for JSON serialization"""
        converted = {}
        for key, value in row.items():
            if value is None:
                converted[key] = None
            elif hasattr(value, '__class__') and 'Decimal' in str(value.__class__):
                # Handle Decimal objects
                converted[key] = float(value)
            elif isinstance(value, (int, float, str, bool)):
                converted[key] = value
            else:
                # For any other type, try to convert to string as fallback
                converted[key] = str(value) if value is not None else None
        return converted
    
    # PROMOTION FINDER OPERATIONS
    async def get_all_promotions(self, search_filter: Optional[str] = None) -> List[Dict]:
        """Get all products with discounts, optionally filtered by search term"""
        query = """
        SELECT product_name, store_name, current_price, regular_price, 
            discount_percentage, ai_main_category, ai_subcategory,
            ai_health_score, ai_nutrition_grade
        FROM Akcije
        WHERE current_price > 0
        """
        params = []
        
        if search_filter:
            query += " AND product_name LIKE %s"
            params.append(f"%{search_filter}%")
        
        query += " ORDER BY discount_percentage DESC, current_price ASC"
        
        results = await self.execute_query(query, params)
        logger.info(f"🏷️ Found {len(results)} promotional items")
        return results
    
    # ITEM FINDER OPERATIONS
    async def find_item_across_stores(self, item_name: str) -> List[Dict]:
        """Find specific item variations across all stores for price comparison"""
        # Use LLM to generate smart search terms with Slovenian support
        search_terms = await self._generate_item_search_terms(item_name)
        
        all_results = []
        for term in search_terms:
            query = """
            SELECT product_name, store_name, current_price, regular_price, 
                   has_discount, discount_percentage, ai_main_category
            FROM Akcije 
            WHERE product_name LIKE %s AND current_price > 0
            ORDER BY current_price ASC
            """
            results = await self.execute_query(query, [f"%{term}%"])
            all_results.extend(results)
        
        # Remove duplicates based on product_name and store_name
        seen = set()
        unique_results = []
        for item in all_results:
            key = (item['product_name'], item['store_name'])
            if key not in seen:
                seen.add(key)
                unique_results.append(item)
        
        logger.info(f"🔍 Found {len(unique_results)} variations of '{item_name}' across stores")
        return unique_results
    
    async def _generate_item_search_terms(self, item_name: str) -> List[str]:
        """Use LLM to generate smart search terms for an item with Slovenian support"""
        try:
            prompt = self.llm_prompts["ingredient_search"].format(search_term=item_name)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=400
                )
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response with better error handling
            import json
            try:
                if "```json" in result_text:
                    json_text = result_text.split("```json")[1].split("```")[0].strip()
                elif result_text.startswith('{'):
                    json_text = result_text
                else:
                    # Try to extract JSON from the response
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_text = result_text[start_idx:end_idx]
                    else:
                        logger.error(f"No JSON found in response: {result_text}")
                        return self._get_fallback_search_terms(item_name)
                
                # Clean up any potential formatting issues
                json_text = json_text.replace('\n', ' ').replace('\r', ' ').strip()
                
                search_data = json.loads(json_text)
                search_terms = search_data.get("search_terms", [item_name])
                
                # Ensure search_terms is a list and contains valid strings
                if not isinstance(search_terms, list):
                    search_terms = [item_name]
                
                # Filter out empty strings and ensure all are strings
                search_terms = [str(term).strip() for term in search_terms if term and str(term).strip()]
                
                if not search_terms:
                    search_terms = self._get_fallback_search_terms(item_name)
                
                # Limit to reasonable number of search terms
                search_terms = search_terms[:7]  # Increased for Slovenian variations
                
                logger.info(f"🧠 Generated Slovenian search terms for '{item_name}': {search_terms}")
                return search_terms
                
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for '{item_name}': {json_error}")
                logger.error(f"Raw response: {result_text}")
                return self._get_fallback_search_terms(item_name)
            
        except Exception as e:
            logger.error(f"LLM search term generation failed: {e}")
            return self._get_fallback_search_terms(item_name)
    
    def _get_fallback_search_terms(self, item_name: str) -> List[str]:
        """Generate fallback search terms using manual Slovenian mappings"""
        # Manual mapping for common items
        fallback_mappings = {
            "milk": ["mleko", "polnomastno mleko", "delno posneto mleko"],
            "bread": ["kruh", "bel kruh", "črn kruh", "toast"],
            "cheese": ["sir", "trdi sir", "mehki sir"],
            "chicken": ["piščanec", "piščančje meso", "piščančji file"],
            "beef": ["goveje", "goveje meso", "govedina"],
            "pork": ["svinjina", "svinjsko meso"],
            "fish": ["riba", "ribje meso"],
            "eggs": ["jajca", "jajce", "kokošja jajca"],
            "butter": ["maslo", "sveto maslo"],
            "yogurt": ["jogurt", "naravni jogurt"],
            "water": ["voda", "mineralna voda"],
            "juice": ["sok", "sadni sok"],
            "coffee": ["kava", "zrnata kava"],
            "tea": ["čaj", "zeliščni čaj"],
            "rice": ["riž", "dolgi riž", "kratki riž"],
            "pasta": ["testenine", "špageti", "makaroni"],
            "potato": ["krompir", "krompirji"],
            "tomato": ["paradižnik", "paradižniki"],
            "onion": ["čebula", "rumena čebula"],
            "apple": ["jabolko", "jabolka"],
            "banana": ["banana", "banane"]
        }
        
        item_lower = item_name.lower()
        
        # Check if the item is in our fallback mappings
        for english_term, slovenian_terms in fallback_mappings.items():
            if english_term in item_lower:
                return slovenian_terms + [item_name]
        
        # Check if it's already a Slovenian term
        for slovenian_terms in fallback_mappings.values():
            for slovenian_term in slovenian_terms:
                if slovenian_term in item_lower:
                    return slovenian_terms + [item_name]
        
        # Default fallback with basic variations
        return [item_name, item_name.lower(), item_name.capitalize()]
    
    # MEAL INGREDIENT OPERATIONS
    async def find_meal_ingredients(self, ingredients: List[str]) -> Dict[str, List[Dict]]:
        """Find prices for meal ingredients across stores with Slovenian support"""
        ingredient_results = {}
        
        for ingredient in ingredients:
            # Generate search terms for this ingredient with Slovenian support
            search_terms = await self._generate_item_search_terms(ingredient)
            
            ingredient_products = []
            for term in search_terms:
                query = """
                SELECT product_name, store_name, current_price, regular_price, 
                       has_discount, ai_main_category, ai_health_score
                FROM Akcije 
                WHERE product_name LIKE %s AND current_price > 0
                ORDER BY current_price ASC
                LIMIT 12
                """
                results = await self.execute_query(query, [f"%{term}%"])
                ingredient_products.extend(results)
            
            # Remove duplicates and keep best matches
            seen = set()
            unique_products = []
            for product in ingredient_products:
                key = (product['product_name'], product['store_name'])
                if key not in seen:
                    seen.add(key)
                    unique_products.append(product)
            
            ingredient_results[ingredient] = unique_products[:8]  # Top 8 per ingredient
        
        logger.info(f"🛒 Found ingredients for meal: {len(ingredient_results)} ingredients processed")
        return ingredient_results
    
    # REVERSE MEAL SEARCH OPERATIONS 
    # TODO TO JE NAROBE
    async def find_meals_by_available_ingredients(self, available_ingredients: List[str]) -> List[Dict]:
        """Find what meals can be made with available ingredients"""

        matching_products = []
        for ingredient in available_ingredients:
            query = """
            SELECT DISTINCT ai_main_category, COUNT(*) as product_count,
                   AVG(current_price) as avg_price
            FROM unified_products_view 
            WHERE product_name LIKE %s AND current_price > 0
            GROUP BY ai_main_category
            ORDER BY product_count DESC
            """
            results = await self.execute_query(query, [f"%{ingredient}%"])
            matching_products.extend(results)
        
        return matching_products
    
    # LLM-POWERED ANALYSIS METHODS
    async def analyze_promotions(self, promotions: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze promotion patterns and provide insights in Slovenian"""
        try:
            prompt = self.llm_prompts["promotion_analysis"].format(
                products=str(promotions[:20])  # Limit for token management
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=600
                )
            )
            
            analysis = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "analysis": analysis,
                "total_promotions": len(promotions)
            }
            
        except Exception as e:
            logger.error(f"Promotion analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Analiza trenutno ni na voljo"
            }
    
    async def analyze_price_comparison(self, price_data: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze price differences and provide recommendations in Slovenian"""
        try:
            prompt = self.llm_prompts["price_comparison"].format(
                price_data=str(price_data)
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Also provide structured data
            if price_data:
                cheapest = min(price_data, key=lambda x: x.get('current_price', float('inf')))
                most_expensive = max(price_data, key=lambda x: x.get('current_price', 0))
                
                structured_analysis = {
                    "cheapest_option": cheapest,
                    "most_expensive": most_expensive,
                    "price_range": {
                        "min": cheapest.get('current_price', 0),
                        "max": most_expensive.get('current_price', 0),
                        "difference": most_expensive.get('current_price', 0) - cheapest.get('current_price', 0)
                    },
                    "stores_compared": len(set(item.get('store_name', '') for item in price_data))
                }
            else:
                structured_analysis = {}
            
            return {
                "success": True,
                "llm_analysis": analysis,
                "structured_data": structured_analysis
            }
            
        except Exception as e:
            logger.error(f"Price comparison analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "structured_data": {}
            }

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

# Global database handler instance
db_handler = None

async def get_db_handler() -> DatabaseHandler:
    """Get or create database handler instance"""
    global db_handler
    if db_handler is None:
        db_handler = DatabaseHandler(get_database_config())
        await db_handler.connect()
    return db_handler

async def close_db_handler():
    """Close database handler connection"""
    global db_handler
    if db_handler:
        db_handler.disconnect()
        db_handler = None