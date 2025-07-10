#!/usr/bin/env python3
"""
Enhanced Search System with Semantic Validation
Prevents wrong products from being returned by validating semantic meaning
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SemanticSearchValidator:
    """Validates search results using AI to ensure semantic correctness"""
    
    def __init__(self):
        self.category_mappings = {
            # Slovenian search terms -> expected categories
            "mleko": ["Mlečni izdelki", "Beverages", "Dairy", "Milk"],
            "kruh": ["Pekovski izdelki", "Bakery", "Bread", "Baked Goods"],
            "jajca": ["Dairy", "Eggs", "Protein", "Fresh"],
            "sir": ["Mlečni izdelki", "Dairy", "Cheese"],
            "meso": ["Meso", "Meat", "Protein", "Fresh"],
            "riba": ["Seafood", "Fish", "Protein", "Fresh"],
            "sadje": ["Sadje", "Fruit", "Fresh", "Produce"],
            "zelenjava": ["Zelenjava", "Vegetables", "Fresh", "Produce"],
            "testenine": ["Testenine", "Pasta", "Grain Products"],
            "riž": ["Grain Products", "Rice", "Dry Goods"],
            "kava": ["Beverages", "Coffee", "Hot Drinks"],
            "čaj": ["Beverages", "Tea", "Hot Drinks"],
            "pivo": ["Beverages", "Beer", "Alcohol"],
            "vino": ["Beverages", "Wine", "Alcohol"],
            "jogurt": ["Mlečni izdelki", "Dairy", "Yogurt"],
            "maslo": ["Mlečni izdelki", "Dairy", "Butter", "Fats"],
            "olje": ["Maščobe", "Oil", "Fats", "Cooking"],
            "sladkor": ["Sweet Products", "Sugar", "Baking"],
            "sol": ["Spices", "Salt", "Seasonings"],
            "krompir": ["Zelenjava", "Vegetables", "Potatoes"],
            "čebula": ["Zelenjava", "Vegetables", "Onions"],
            "paradižnik": ["Zelenjava", "Vegetables", "Tomatoes"],
            "jabolka": ["Sadje", "Fruit", "Apples"],
            "banane": ["Sadje", "Fruit", "Bananas"],
            "čokolada": ["Sweet Products", "Chocolate", "Confectionery"],
        }
        
        self.brand_exclusions = {
            # When searching for these terms, exclude these brands/words
            "mleko": ["milka", "milky", "milfina"],  # Exclude chocolate brands
            "kruh": ["breadcrumbs", "bread crumbs"],  # Exclude breadcrumbs when looking for bread
        }

    async def validate_search_results(
        self, 
        search_term: str, 
        results: List[Dict], 
        max_results: int = 10
    ) -> List[Dict]:
        """
        Validate search results to ensure they semantically match the search intent
        """
        if not results:
            return results
        
        logger.info(f"🔍 Validating {len(results)} results for search term '{search_term}'")
        
        # Step 1: Quick category-based filtering
        category_filtered = self._filter_by_category(search_term, results)
        logger.info(f"📂 Category filtering: {len(category_filtered)} results remain")
        
        # Step 2: Brand/word exclusion filtering
        brand_filtered = self._filter_by_brand_exclusions(search_term, category_filtered)
        logger.info(f"🚫 Brand exclusion filtering: {len(brand_filtered)} results remain")
        
        # Step 3: AI semantic validation for remaining results
        if len(brand_filtered) > 0:
            ai_validated = await self._ai_semantic_validation(search_term, brand_filtered, max_results)
            logger.info(f"🤖 AI validation: {len(ai_validated)} results remain")
            return ai_validated
        
        return brand_filtered[:max_results]

    def _filter_by_category(self, search_term: str, results: List[Dict]) -> List[Dict]:
        """Filter results based on expected categories for the search term"""
        search_term_lower = search_term.lower().strip()
        expected_categories = self.category_mappings.get(search_term_lower, [])
        
        if not expected_categories:
            # If we don't have category mapping, return all results
            return results
        
        filtered_results = []
        
        for result in results:
            category = result.get('ai_main_category', '').lower()
            subcategory = result.get('ai_subcategory', '').lower()
            
            # Check if the product's category matches expected categories
            category_match = any(
                expected.lower() in category or expected.lower() in subcategory
                for expected in expected_categories
            )
            
            if category_match:
                filtered_results.append(result)
            else:
                logger.debug(f"❌ Category mismatch: '{result.get('product_name')}' "
                           f"(category: {category}) doesn't match expected categories for '{search_term}'")
        
        return filtered_results

    def _filter_by_brand_exclusions(self, search_term: str, results: List[Dict]) -> List[Dict]:
        """Filter out products that contain excluded brand names or words"""
        search_term_lower = search_term.lower().strip()
        exclusions = self.brand_exclusions.get(search_term_lower, [])
        
        if not exclusions:
            return results
        
        filtered_results = []
        
        for result in results:
            product_name = result.get('product_name', '').lower()
            
            # Check if product name contains any excluded terms
            contains_exclusion = any(exclusion in product_name for exclusion in exclusions)
            
            if not contains_exclusion:
                filtered_results.append(result)
            else:
                logger.debug(f"🚫 Brand exclusion: '{result.get('product_name')}' "
                           f"contains excluded term for '{search_term}'")
        
        return filtered_results

    async def _ai_semantic_validation(
        self, 
        search_term: str, 
        results: List[Dict], 
        max_results: int
    ) -> List[Dict]:
        """Use AI to validate if products semantically match the search intent"""
        
        if not results:
            return results
        
        # Take top candidates for AI validation (don't validate all for performance)
        candidates = results[:max_results * 2]  # Get more candidates than needed
        
        # Create prompt for AI validation
        products_text = "\n".join([
            f"{i+1}. {result['product_name']} (Category: {result.get('ai_main_category', 'Unknown')})"
            for i, result in enumerate(candidates)
        ])
        
        validation_prompt = f"""
You are a grocery product validator. A user searched for "{search_term}" in Slovenian.
Here are the product candidates found:

{products_text}

Your task: Identify which products actually match what the user is looking for when they search for "{search_term}".

For example:
- If searching for "mleko" (milk), exclude chocolate products like "MLEČNA REZINA MILKA" even if they contain similar words
- If searching for "kruh" (bread), exclude breadcrumbs or bread-related items that aren't actual bread
- If searching for "jabolka" (apples), exclude apple juice or apple-flavored products unless they're actual apples

Return ONLY a JSON array of numbers representing the products that are valid matches (1-based indexing).
For example: [1, 3, 5] means products 1, 3, and 5 are valid matches.

If no products are valid matches, return an empty array: []

Response format: Just the JSON array, nothing else.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=100
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"🤖 AI validation response: {ai_response}")
            
            # Extract valid indices
            valid_indices = json.loads(ai_response)
            
            # Return only the valid products
            validated_results = []
            for index in valid_indices:
                if 1 <= index <= len(candidates):
                    validated_results.append(candidates[index - 1])
            
            # Limit to max_results
            return validated_results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ AI validation failed: {e}")
            # Fallback: return first few results if AI validation fails
            return results[:max_results]

# Enhanced Search Functions for the main system

class EnhancedProductSearch:
    """Enhanced product search with semantic validation"""
    
    def __init__(self, grocery_mcp, db_source):
        self.grocery_mcp = grocery_mcp
        self.db_source = db_source
        self.validator = SemanticSearchValidator()

    async def find_cheapest_product_validated(
        self, 
        product_name: str, 
        location: str = "Ljubljana", 
        store_preference: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """Find cheapest product with semantic validation"""
        
        # Step 1: Get initial search results (more than we need)
        initial_results = await self.grocery_mcp.find_cheapest_product(
            product_name, location, store_preference
        )
        
        if not initial_results:
            logger.info(f"❌ No initial results found for '{product_name}'")
            return []
        
        # Step 2: Apply semantic validation
        validated_results = await self.validator.validate_search_results(
            product_name, initial_results, max_results
        )
        
        if not validated_results:
            logger.warning(f"⚠️ No products passed semantic validation for '{product_name}'")
            # Optionally return a message about expanding search or trying different terms
            return []
        
        logger.info(f"✅ Returning {len(validated_results)} validated results for '{product_name}'")
        return validated_results

    async def search_with_suggestions(
        self, 
        product_name: str, 
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Enhanced search that provides suggestions if no valid results found
        """
        validated_results = await self.find_cheapest_product_validated(
            product_name, max_results=max_results
        )
        
        if validated_results:
            return {
                "success": True,
                "results": validated_results,
                "message": f"Found {len(validated_results)} products matching '{product_name}'"
            }
        
        # If no validated results, try to provide helpful suggestions
        suggestions = await self._generate_search_suggestions(product_name)
        
        return {
            "success": False,
            "results": [],
            "message": f"No products found for '{product_name}'",
            "suggestions": suggestions
        }

    async def _generate_search_suggestions(self, product_name: str) -> List[str]:
        """Generate search suggestions when no results found"""
        
        # Common alternative terms in Slovenian
        suggestions_map = {
            "milk": ["mleko", "mlečni izdelki"],
            "bread": ["kruh", "pekovski izdelki"],
            "eggs": ["jajca"],
            "cheese": ["sir", "mlečni izdelki"],
            "meat": ["meso"],
            "fish": ["riba"],
            "apple": ["jabolka", "sadje"],
            "coffee": ["kava"],
            "tea": ["čaj"],
            "pasta": ["testenine"],
            "rice": ["riž"]
        }
        
        # Try to find suggestions based on the search term
        product_lower = product_name.lower()
        suggestions = []
        
        for english_term, slovenian_terms in suggestions_map.items():
            if english_term in product_lower or any(term in product_lower for term in slovenian_terms):
                suggestions.extend(slovenian_terms)
        
        # Remove duplicates and the original search term
        suggestions = list(set(suggestions))
        if product_name.lower() in [s.lower() for s in suggestions]:
            suggestions = [s for s in suggestions if s.lower() != product_name.lower()]
        
        return suggestions[:5]  # Return top 5 suggestions

# Integration functions for the existing system

async def enhanced_find_cheapest_product(
    product_name: str,
    grocery_mcp,
    db_source,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced find_cheapest_product function that can be used in the existing system
    """
    search_engine = EnhancedProductSearch(grocery_mcp, db_source)
    
    # Use the validated search
    result = await search_engine.search_with_suggestions(product_name)
    
    if result["success"]:
        return {
            "products": result["results"],
            "message": result["message"],
            "validation_applied": True
        }
    else:
        return {
            "products": [],
            "message": result["message"],
            "suggestions": result.get("suggestions", []),
            "validation_applied": True
        }

# Example usage and testing
async def test_semantic_validation():
    """Test the semantic validation system"""
    
    # Mock results for testing
    mock_results = [
        {
            "product_name": "MLEKO UHT 3,5% MAŠČOBE 1L",
            "store_name": "mercator",
            "current_price": 1.19,
            "ai_main_category": "Mlečni izdelki"
        },
        {
            "product_name": "MLEČNA REZINA MILKA 29 G",
            "store_name": "spar", 
            "current_price": 0.58,
            "ai_main_category": "Sladkarije"
        },
        {
            "product_name": "POLNOMASTNO MLEKO 1L",
            "store_name": "lidl",
            "current_price": 0.89,
            "ai_main_category": "Mlečni izdelki"
        }
    ]
    
    validator = SemanticSearchValidator()
    
    print("🧪 Testing semantic validation for 'mleko':")
    validated = await validator.validate_search_results("mleko", mock_results)
    
    print(f"📊 Original results: {len(mock_results)}")
    print(f"✅ Validated results: {len(validated)}")
    
    for result in validated:
        print(f"   - {result['product_name']} ({result['ai_main_category']})")

if __name__ == "__main__":
    asyncio.run(test_semantic_validation())