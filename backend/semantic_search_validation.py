#!/usr/bin/env python3
"""
Dynamic LLM-Based Semantic Search Validation
Uses the LLM to understand database content and determine product relevance
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

class DynamicSemanticValidator:
    """
    Dynamic semantic validator that uses LLM to understand database content
    and determine product relevance without hard-coded rules
    """
    
    def __init__(self):
        self.client = client
        
    async def validate_search_results(
        self, 
        search_term: str, 
        raw_results: List[Dict], 
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Validate search results using LLM understanding of database content
        
        Returns:
        {
            "valid_products": List[Dict],
            "invalid_products": List[Dict],
            "reasoning": str,
            "suggestions": List[str],
            "confidence": float
        }
        """
        
        if not raw_results:
            return {
                "valid_products": [],
                "invalid_products": [],
                "reasoning": "No products found in database",
                "suggestions": [],
                "confidence": 0.0
            }
        
        logger.info(f"🤖 Validating {len(raw_results)} products for search term '{search_term}'")
        
        # Step 1: Analyze the database content first
        database_analysis = await self._analyze_database_content(search_term, raw_results)
        
        # Step 2: Use LLM to determine product relevance
        validation_result = await self._llm_validate_products(
            search_term, 
            raw_results, 
            database_analysis,
            max_results
        )
        
        # Step 3: Generate suggestions if needed
        if not validation_result["valid_products"]:
            suggestions = await self._generate_intelligent_suggestions(
                search_term, 
                raw_results, 
                database_analysis
            )
            validation_result["suggestions"] = suggestions
        
        return validation_result
    
    async def _analyze_database_content(
        self, 
        search_term: str, 
        raw_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze what's actually in the database for this search term
        """
        
        # Extract product information for analysis
        product_info = []
        categories = set()
        stores = set()
        
        for result in raw_results[:20]:  # Analyze first 20 results
            product_info.append({
                "name": result.get("product_name", ""),
                "category": result.get("ai_main_category", ""),
                "subcategory": result.get("ai_subcategory", ""),
                "store": result.get("store_name", ""),
                "price": result.get("current_price", 0)
            })
            
            if result.get("ai_main_category"):
                categories.add(result.get("ai_main_category"))
            if result.get("store_name"):
                stores.add(result.get("store_name"))
        
        # Ask LLM to analyze the database content
        analysis_prompt = f"""
        You are analyzing a grocery database search. A user searched for "{search_term}" and these are the products found:

        PRODUCTS FOUND:
        {json.dumps(product_info, indent=2)}

        CATEGORIES FOUND: {', '.join(categories)}
        STORES: {', '.join(stores)}

        Please analyze:
        1. What types of products are actually in the database results?
        2. What was the user most likely looking for when they searched for "{search_term}"?
        3. Are there clear patterns in the product names or categories?
        4. Are there any obvious mismatches (e.g., chocolate products when searching for milk)?

        Respond with a JSON object containing:
        {{
            "user_intent": "What the user was probably looking for",
            "database_content_summary": "Summary of what's actually in the database",
            "potential_issues": ["List of potential mismatches"],
            "dominant_categories": ["Most common categories found"],
            "analysis_confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                if "```json" in analysis_text:
                    json_text = analysis_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_text = analysis_text
                    
                analysis = json.loads(json_text)
                logger.info(f"🧠 Database analysis: {analysis.get('user_intent', 'Unknown intent')}")
                return analysis
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse analysis JSON, using fallback")
                return {
                    "user_intent": f"Products related to '{search_term}'",
                    "database_content_summary": f"Found {len(product_info)} products",
                    "potential_issues": [],
                    "dominant_categories": list(categories),
                    "analysis_confidence": 0.5
                }
                
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            return {
                "user_intent": f"Products related to '{search_term}'",
                "database_content_summary": f"Found {len(product_info)} products",
                "potential_issues": [],
                "dominant_categories": list(categories),
                "analysis_confidence": 0.3
            }
    
    async def _llm_validate_products(
        self,
        search_term: str,
        raw_results: List[Dict],
        database_analysis: Dict[str, Any],
        max_results: int
    ) -> Dict[str, Any]:
        """
        Use LLM to validate which products actually match the user's search intent
        """
        
        # Prepare products for validation (limit to prevent token overflow)
        validation_products = []
        for i, result in enumerate(raw_results[:30]):  # Validate up to 30 products
            validation_products.append({
                "index": i,
                "name": result.get("product_name", ""),
                "category": result.get("ai_main_category", ""),
                "subcategory": result.get("ai_subcategory", ""),
                "store": result.get("store_name", ""),
                "price": result.get("current_price", 0),
                "summary": result.get("ai_product_summary", "")
            })
        
        validation_prompt = f"""
        You are a grocery shopping assistant with expertise in product categorization.

        USER SEARCH: "{search_term}"
        USER INTENT: {database_analysis.get('user_intent', 'Unknown')}
        
        DATABASE ANALYSIS:
        - Content Summary: {database_analysis.get('database_content_summary', '')}
        - Potential Issues: {database_analysis.get('potential_issues', [])}
        - Dominant Categories: {database_analysis.get('dominant_categories', [])}

        PRODUCTS TO VALIDATE:
        {json.dumps(validation_products, indent=2)}

        Your task: Determine which products actually match what the user is looking for.

        Consider:
        1. Does the product name match the user's intent?
        2. Is the category appropriate for what they're searching for?
        3. Are there obvious mismatches (e.g., chocolate when searching for milk)?
        4. Would a typical shopper expect this product when searching for "{search_term}"?

        Respond with JSON:
        {{
            "valid_indices": [0, 2, 5],  // Array of indices for valid products
            "invalid_indices": [1, 3, 4],  // Array of indices for invalid products
            "reasoning": "Explanation of validation decisions",
            "confidence": 0.0-1.0,
            "validation_summary": "Brief summary of what was accepted/rejected"
        }}

        Be reasonable - don't be too strict, but filter out obvious mismatches.
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": validation_prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
            )
            
            validation_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                if "```json" in validation_text:
                    json_text = validation_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_text = validation_text
                    
                validation_result = json.loads(json_text)
                
                # Build final results
                valid_products = []
                invalid_products = []
                
                for idx in validation_result.get("valid_indices", []):
                    if 0 <= idx < len(raw_results):
                        valid_products.append(raw_results[idx])
                
                for idx in validation_result.get("invalid_indices", []):
                    if 0 <= idx < len(raw_results):
                        invalid_products.append(raw_results[idx])
                
                # Limit to max_results
                valid_products = valid_products[:max_results]
                
                logger.info(f"✅ Validation: {len(valid_products)} valid, {len(invalid_products)} invalid")
                logger.info(f"🧠 Reasoning: {validation_result.get('reasoning', 'No reasoning provided')}")
                
                return {
                    "valid_products": valid_products,
                    "invalid_products": invalid_products,
                    "reasoning": validation_result.get("reasoning", ""),
                    "confidence": validation_result.get("confidence", 0.5),
                    "validation_summary": validation_result.get("validation_summary", ""),
                    "suggestions": []
                }
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse validation JSON, using fallback")
                # Fallback: return first max_results products
                return {
                    "valid_products": raw_results[:max_results],
                    "invalid_products": [],
                    "reasoning": "Validation parsing failed, returned all products",
                    "confidence": 0.3,
                    "validation_summary": "Fallback validation",
                    "suggestions": []
                }
                
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Fallback: return first max_results products
            return {
                "valid_products": raw_results[:max_results],
                "invalid_products": [],
                "reasoning": f"Validation failed: {str(e)}",
                "confidence": 0.2,
                "validation_summary": "Error fallback",
                "suggestions": []
            }
    
    async def _generate_intelligent_suggestions(
        self,
        search_term: str,
        raw_results: List[Dict],
        database_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate intelligent search suggestions based on database content
        """
        
        # Extract categories and product patterns from database
        categories = set()
        product_patterns = []
        
        for result in raw_results[:10]:
            if result.get("ai_main_category"):
                categories.add(result.get("ai_main_category"))
            
            # Extract key words from product names
            product_name = result.get("product_name", "").lower()
            words = product_name.split()
            for word in words:
                if len(word) > 3:  # Only meaningful words
                    product_patterns.append(word)
        
        suggestion_prompt = f"""
        A user searched for "{search_term}" but no valid products were found.

        DATABASE ANALYSIS:
        - User Intent: {database_analysis.get('user_intent', 'Unknown')}
        - Found Categories: {', '.join(categories)}
        - Potential Issues: {database_analysis.get('potential_issues', [])}

        SAMPLE PRODUCT WORDS FROM DATABASE: {', '.join(list(set(product_patterns))[:20])}

        Generate 3-5 alternative search terms that might help the user find what they're looking for.
        Consider:
        1. Different ways to express the same product in Slovenian
        2. More general or specific terms
        3. Alternative product names or brands
        4. Related products

        Respond with JSON:
        {{
            "suggestions": ["term1", "term2", "term3"],
            "reasoning": "Why these suggestions might work better"
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": suggestion_prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
            )
            
            suggestion_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                if "```json" in suggestion_text:
                    json_text = suggestion_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_text = suggestion_text
                    
                suggestion_result = json.loads(json_text)
                suggestions = suggestion_result.get("suggestions", [])
                
                logger.info(f"💡 Generated {len(suggestions)} suggestions for '{search_term}'")
                return suggestions[:5]  # Limit to 5 suggestions
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse suggestions JSON")
                return []
                
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []

# Enhanced integration for the existing system
class EnhancedProductSearchWithDynamicValidation:
    """
    Enhanced product search with dynamic LLM-based validation
    """
    
    def __init__(self, grocery_mcp, db_source):
        self.grocery_mcp = grocery_mcp
        self.db_source = db_source
        self.validator = DynamicSemanticValidator()

    async def search_products_with_intelligent_validation(
        self, 
        search_term: str, 
        max_results: int = 10,
        validation_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Search products with intelligent LLM-based validation
        """
        
        # Step 1: Get raw database results
        raw_results = await self._get_raw_database_results(search_term, limit=50)
        
        if not raw_results:
            return {
                "success": False,
                "products": [],
                "message": f"No products found for '{search_term}'",
                "search_term": search_term,
                "validation_applied": False,
                "suggestions": []
            }
        
        logger.info(f"🔍 Found {len(raw_results)} raw results for '{search_term}'")
        
        # Step 2: Apply dynamic validation if enabled
        if validation_enabled:
            validation_result = await self.validator.validate_search_results(
                search_term, raw_results, max_results
            )
            
            if validation_result["valid_products"]:
                return {
                    "success": True,
                    "products": validation_result["valid_products"],
                    "message": f"Found {len(validation_result['valid_products'])} validated products for '{search_term}'",
                    "search_term": search_term,
                    "validation_applied": True,
                    "validation_reasoning": validation_result["reasoning"],
                    "validation_confidence": validation_result["confidence"],
                    "invalid_products_count": len(validation_result["invalid_products"]),
                    "suggestions": []
                }
            else:
                # No valid products found
                return {
                    "success": False,
                    "products": [],
                    "message": f"No valid products found for '{search_term}' after validation",
                    "search_term": search_term,
                    "validation_applied": True,
                    "validation_reasoning": validation_result["reasoning"],
                    "validation_confidence": validation_result["confidence"],
                    "suggestions": validation_result["suggestions"],
                    "raw_results_count": len(raw_results)
                }
        
        # Step 3: Return raw results if validation disabled
        formatted_results = raw_results[:max_results]
        return {
            "success": True,
            "products": formatted_results,
            "message": f"Found {len(formatted_results)} products for '{search_term}' (no validation)",
            "search_term": search_term,
            "validation_applied": False,
            "suggestions": []
        }

    async def _get_raw_database_results(
        self, 
        search_term: str, 
        limit: int = 50
    ) -> List[Dict]:
        """
        Get raw search results from database
        """
        try:
            # Use the existing grocery MCP search
            results = await self.grocery_mcp.find_cheapest_product(
                search_term, use_semantic_validation=False
            )
            return results[:limit]
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []

# Integration function for existing system
async def enhanced_search_with_dynamic_validation(
    search_term: str,
    grocery_mcp,
    db_source,
    max_results: int = 10,
    validation_enabled: bool = True
) -> Dict[str, Any]:
    """
    Enhanced search function that can be used in the existing system
    """
    search_engine = EnhancedProductSearchWithDynamicValidation(grocery_mcp, db_source)
    
    result = await search_engine.search_products_with_intelligent_validation(
        search_term, max_results, validation_enabled
    )
    
    return result

# Testing function
async def test_dynamic_validation():
    """
    Test the dynamic validation system
    """
    
    # Mock data for testing
    mock_results_mleko = [
        {
            "product_name": "MLEKO UHT 3,5% MAŠČOBE 1L",
            "store_name": "mercator",
            "current_price": 1.19,
            "ai_main_category": "Mlečni izdelki",
            "ai_subcategory": "Mleko",
            "ai_product_summary": "UHT milk with 3.5% fat content"
        },
        {
            "product_name": "MLEČNA REZINA MILKA 28G",
            "store_name": "spar", 
            "current_price": 0.58,
            "ai_main_category": "Sladkarije",
            "ai_subcategory": "Čokolada",
            "ai_product_summary": "Chocolate bar with milk filling"
        },
        {
            "product_name": "POLNOMASTNO MLEKO 1L",
            "store_name": "lidl",
            "current_price": 0.89,
            "ai_main_category": "Mlečni izdelki",
            "ai_subcategory": "Mleko",
            "ai_product_summary": "Full-fat fresh milk"
        },
        {
            "product_name": "MILKA ČOKOLADA OREO 100G",
            "store_name": "dm",
            "current_price": 1.49,
            "ai_main_category": "Sladkarije",
            "ai_subcategory": "Čokolada",
            "ai_product_summary": "Chocolate bar with Oreo cookies"
        }
    ]
    
    validator = DynamicSemanticValidator()
    
    print("🧪 Testing Dynamic Validation for 'mleko':")
    result = await validator.validate_search_results("mleko", mock_results_mleko)
    
    print(f"✅ Valid products: {len(result['valid_products'])}")
    for product in result['valid_products']:
        print(f"   - {product['product_name']} ({product['ai_main_category']})")
    
    print(f"❌ Invalid products: {len(result['invalid_products'])}")
    for product in result['invalid_products']:
        print(f"   - {product['product_name']} ({product['ai_main_category']})")
    
    print(f"🧠 Reasoning: {result['reasoning']}")
    print(f"📊 Confidence: {result['confidence']:.2f}")
    
    if result['suggestions']:
        print(f"💡 Suggestions: {', '.join(result['suggestions'])}")

if __name__ == "__main__":
    asyncio.run(test_dynamic_validation())