#!/usr/bin/env python3
"""
Item Finder Module
Finds specific items across all stores for price comparison
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from database_handler import get_db_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ItemFinder:
    """
    Handles finding and comparing items across stores
    """
    
    def __init__(self):
        self.db_handler = None
        self.stores = ["dm", "lidl", "mercator", "spar", "tus"]
    
    async def _ensure_db_connection(self):
        """Ensure database connection is available"""
        if self.db_handler is None:
            self.db_handler = await get_db_handler()
    
    async def compare_item_prices(
        self,
        item_name: str,
        include_similar: bool = True,
        max_results_per_store: int = 5
    ) -> Dict[str, Any]:
        """
        Find and compare prices for a specific item across all stores
        
        Args:
            item_name: Name of the item to search for
            include_similar: Whether to include similar/related products
            max_results_per_store: Maximum results per store
        """
        await self._ensure_db_connection()
        
        logger.info(f"🔍 Comparing prices for item: '{item_name}'")
        
        try:
            # Find all variations of the item
            all_results = await self.db_handler.find_item_across_stores(item_name)
            
            if not all_results:
                return {
                    "success": False,
                    "message": f"No items found matching '{item_name}'",
                    "item_searched": item_name,
                    "suggestions": await self._get_search_suggestions(item_name)
                }
            
            # Process and organize results
            organized_results = self._organize_by_store(all_results, max_results_per_store)
            
            # Generate price analysis
            price_analysis = await self._analyze_price_comparison(all_results)
            
            # Find best deals
            best_deals = self._find_best_deals(all_results)
            
            # Calculate store rankings
            store_rankings = self._rank_stores(organized_results)
            
            # Generate product variations analysis
            variations_analysis = self._analyze_product_variations(all_results)
            
            result = {
                "success": True,
                "item_searched": item_name,
                "total_products_found": len(all_results),
                "stores_with_products": len([store for store in organized_results.values() if store["products"]]),
                "results_by_store": organized_results,
                "price_analysis": price_analysis,
                "best_deals": best_deals,
                "store_rankings": store_rankings,
                "variations_analysis": variations_analysis,
                "summary": self._generate_comparison_summary(item_name, all_results, price_analysis)
            }
            
            logger.info(f"✅ Found {len(all_results)} products across {len(organized_results)} stores")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error comparing item prices: {e}")
            return {
                "success": False,
                "error": str(e),
                "item_searched": item_name,
                "message": "Failed to compare prices"
            }
    
    def _organize_by_store(self, results: List[Dict], max_per_store: int) -> Dict[str, Dict]:
        """Organize results by store"""
        organized = {}
        
        # Initialize all stores
        for store in self.stores:
            organized[store] = {
                "store_name": store.upper(),
                "products": [],
                "cheapest_product": None,
                "product_count": 0,
                "avg_price": 0,
                "price_range": {"min": None, "max": None}
            }
        
        # Group products by store
        for product in results:
            store_name = product.get('store_name', '').lower()
            if store_name in self.stores:
                store_data = organized[store_name]
                
                # Add product if under limit
                if len(store_data["products"]) < max_per_store:
                    store_data["products"].append(product)
                
                # Update statistics
                store_data["product_count"] += 1
                current_price = product.get('current_price', 0) or 0
                
                if store_data["cheapest_product"] is None or current_price < store_data["cheapest_product"].get('current_price', float('inf')):
                    store_data["cheapest_product"] = product
                
                # Update price range
                if store_data["price_range"]["min"] is None or current_price < store_data["price_range"]["min"]:
                    store_data["price_range"]["min"] = current_price
                if store_data["price_range"]["max"] is None or current_price > store_data["price_range"]["max"]:
                    store_data["price_range"]["max"] = current_price
        
        # Calculate average prices
        for store_name, store_data in organized.items():
            if store_data["products"]:
                avg_price = sum(p.get('current_price', 0) for p in store_data["products"]) / len(store_data["products"])
                store_data["avg_price"] = round(avg_price, 2)
        
        return organized
    
    async def _analyze_price_comparison(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze price differences and provide insights"""
        if not results:
            return {}
        
        prices = [p.get('current_price', 0) for p in results if p.get('current_price', 0) > 0]
        
        if not prices:
            return {"message": "No valid prices found"}
        
        # Basic price statistics
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        price_difference = max_price - min_price
        
        # Find cheapest and most expensive
        cheapest_product = min(results, key=lambda x: x.get('current_price', float('inf')))
        most_expensive = max(results, key=lambda x: x.get('current_price', 0))
        
        # Calculate potential savings
        potential_savings = price_difference
        savings_percentage = (price_difference / max_price * 100) if max_price > 0 else 0
        
        analysis = {
            "price_statistics": {
                "min_price": round(min_price, 2),
                "max_price": round(max_price, 2),
                "average_price": round(avg_price, 2),
                "price_difference": round(price_difference, 2),
                "price_range_percentage": round(savings_percentage, 1)
            },
            "cheapest_option": {
                "product_name": cheapest_product.get('product_name'),
                "store": cheapest_product.get('store_name', '').upper(),
                "price": cheapest_product.get('current_price'),
                "has_discount": cheapest_product.get('has_discount', False),
                "discount_percentage": cheapest_product.get('discount_percentage')
            },
            "most_expensive": {
                "product_name": most_expensive.get('product_name'),
                "store": most_expensive.get('store_name', '').upper(),
                "price": most_expensive.get('current_price')
            },
            "savings_potential": {
                "max_savings": round(potential_savings, 2),
                "savings_percentage": round(savings_percentage, 1),
                "recommendation": "Always choose the cheapest option" if potential_savings > 1 else "Prices are similar across stores"
            }
        }
        
        # Use LLM for deeper analysis if available
        if self.db_handler:
            try:
                llm_analysis = await self.db_handler.analyze_price_comparison(results[:10])
                if llm_analysis["success"]:
                    analysis["llm_insights"] = llm_analysis["llm_analysis"]
                    analysis["structured_insights"] = llm_analysis["structured_data"]
            except Exception as e:
                logger.warning(f"LLM price analysis failed: {e}")
        
        return analysis
    
    def _find_best_deals(self, results: List[Dict]) -> Dict[str, Any]:
        """Find the best deals among the results"""
        if not results:
            return {}
        
        # Find products with discounts
        discounted_products = [p for p in results if p.get('has_discount', False)]
        
        # Find overall cheapest
        cheapest_overall = min(results, key=lambda x: x.get('current_price', float('inf')))
        
        # Find best discount percentage
        best_discount = max(discounted_products, key=lambda x: x.get('discount_percentage', 0)) if discounted_products else None
        
        # Find best value (considering health score if available)
        best_value = self._calculate_best_value(results)
        
        return {
            "cheapest_overall": {
                "product": cheapest_overall.get('product_name'),
                "store": cheapest_overall.get('store_name', '').upper(),
                "price": cheapest_overall.get('current_price'),
                "savings_vs_expensive": self._calculate_savings_vs_most_expensive(cheapest_overall, results)
            },
            "best_discount": {
                "product": best_discount.get('product_name') if best_discount else None,
                "store": best_discount.get('store_name', '').upper() if best_discount else None,
                "discount": best_discount.get('discount_percentage') if best_discount else None,
                "price": best_discount.get('current_price') if best_discount else None,
                "original_price": best_discount.get('regular_price') if best_discount else None
            } if best_discount else None,
            "best_value": best_value,
            "total_discounted_items": len(discounted_products)
        }
    
    def _calculate_best_value(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate best value considering price and quality factors"""
        if not results:
            return {}
        
        best_value_product = None
        best_value_score = 0
        
        for product in results:
            price = product.get('current_price', 0) or 0
            health_score = product.get('ai_health_score', 5) or 5
            discount = product.get('discount_percentage', 0) or 0
            
            # Calculate value score (higher health score, lower price, higher discount = better value)
            if price > 0:
                price_score = max(0, 1 - (price / 10))  # Normalize price (assuming max reasonable price is 10)
                health_score_norm = health_score / 10
                discount_score = discount / 100
                
                value_score = (health_score_norm * 0.4 + price_score * 0.4 + discount_score * 0.2)
                
                if value_score > best_value_score:
                    best_value_score = value_score
                    best_value_product = product
        
        if best_value_product:
            return {
                "product": best_value_product.get('product_name'),
                "store": best_value_product.get('store_name', '').upper(),
                "price": best_value_product.get('current_price'),
                "health_score": best_value_product.get('ai_health_score'),
                "value_score": round(best_value_score, 3),
                "reasoning": "Best combination of price, health score, and discounts"
            }
        
        return {}
    
    def _calculate_savings_vs_most_expensive(self, cheapest: Dict, all_results: List[Dict]) -> float:
        """Calculate savings compared to most expensive option"""
        if not all_results:
            return 0
        
        max_price = max(p.get('current_price', 0) for p in all_results)
        cheapest_price = cheapest.get('current_price', 0)
        
        return round(max_price - cheapest_price, 2)
    
    def _rank_stores(self, organized_results: Dict[str, Dict]) -> List[Dict]:
        """Rank stores based on price competitiveness"""
        rankings = []
        
        for store_name, store_data in organized_results.items():
            if store_data["products"]:
                score = 0
                factors = []
                
                # Factor 1: Average price (lower is better)
                avg_price = store_data["avg_price"]
                all_avg_prices = [s["avg_price"] for s in organized_results.values() if s["avg_price"] > 0]
                if all_avg_prices:
                    min_avg = min(all_avg_prices)
                    max_avg = max(all_avg_prices)
                    if max_avg > min_avg:
                        price_score = 1 - ((avg_price - min_avg) / (max_avg - min_avg))
                        score += price_score * 0.4
                        factors.append(f"Price competitiveness: {price_score:.2f}")
                
                # Factor 2: Product availability
                availability_score = min(store_data["product_count"] / 10, 1.0)  # Normalize to max 10 products
                score += availability_score * 0.3
                factors.append(f"Product variety: {availability_score:.2f}")
                
                # Factor 3: Has cheapest option
                cheapest_product = store_data["cheapest_product"]
                all_cheapest_prices = [s["cheapest_product"].get('current_price', float('inf')) 
                                     for s in organized_results.values() if s["cheapest_product"]]
                if all_cheapest_prices and cheapest_product:
                    global_cheapest = min(all_cheapest_prices)
                    if cheapest_product.get('current_price', 0) == global_cheapest:
                        score += 0.3
                        factors.append("Has cheapest option: 0.30")
                
                rankings.append({
                    "store": store_name.upper(),
                    "rank_score": round(score, 3),
                    "avg_price": avg_price,
                    "product_count": store_data["product_count"],
                    "cheapest_price": cheapest_product.get('current_price') if cheapest_product else None,
                    "factors": factors
                })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x["rank_score"], reverse=True)
        
        # Add rank numbers
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _analyze_product_variations(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze different product variations found"""
        if not results:
            return {}
        
        # Group by similar product names
        variations = {}
        categories = set()
        
        for product in results:
            product_name = product.get('product_name', '').lower()
            category = product.get('ai_main_category', 'Unknown')
            categories.add(category)
            
            # Simple grouping by first few words
            key_words = ' '.join(product_name.split()[:3])
            
            if key_words not in variations:
                variations[key_words] = {
                    "representative_name": product.get('product_name'),
                    "products": [],
                    "price_range": {"min": float('inf'), "max": 0},
                    "stores": set()
                }
            
            var_data = variations[key_words]
            var_data["products"].append(product)
            var_data["stores"].add(product.get('store_name', ''))
            
            price = product.get('current_price', 0) or 0
            if price < var_data["price_range"]["min"]:
                var_data["price_range"]["min"] = price
            if price > var_data["price_range"]["max"]:
                var_data["price_range"]["max"] = price
        
        # Process variations
        processed_variations = []
        for key, var_data in variations.items():
            var_data["stores"] = list(var_data["stores"])
            var_data["product_count"] = len(var_data["products"])
            var_data["store_count"] = len(var_data["stores"])
            
            # Fix infinite min price
            if var_data["price_range"]["min"] == float('inf'):
                var_data["price_range"]["min"] = 0
            
            processed_variations.append(var_data)
        
        # Sort by product count
        processed_variations.sort(key=lambda x: x["product_count"], reverse=True)
        
        return {
            "total_variations": len(processed_variations),
            "categories_found": list(categories),
            "variations": processed_variations[:10],  # Top 10 variations
            "analysis_summary": f"Found {len(processed_variations)} product variations across {len(categories)} categories"
        }
    
    async def _get_search_suggestions(self, item_name: str) -> List[str]:
        """Get search suggestions for items that weren't found"""
        # Simple suggestions based on common alternatives
        suggestions = []
        
        item_lower = item_name.lower()
        
        # Common alternatives mapping
        alternatives = {
            "milk": ["mleko", "dairy"],
            "bread": ["kruh", "bakery"],
            "cheese": ["sir", "dairy"],
            "meat": ["meso", "beef", "chicken", "pork"],
            "fish": ["riba", "seafood"],
            "apple": ["jabolko", "fruit"],
            "potato": ["krompir", "vegetables"]
        }
        
        for key, alts in alternatives.items():
            if key in item_lower:
                suggestions.extend(alts)
        
        # Add generic suggestions
        suggestions.extend([
            "Try using Slovenian terms",
            "Use more general terms (e.g., 'dairy' instead of specific brands)",
            "Check spelling"
        ])
        
        return suggestions[:5]
    
    def _generate_comparison_summary(self, item_name: str, results: List[Dict], price_analysis: Dict) -> str:
        """Generate a human-readable summary of the price comparison"""
        if not results:
            return f"No products found for '{item_name}'. Try different search terms."
        
        total_products = len(results)
        stores_count = len(set(p.get('store_name', '') for p in results))
        
        summary = f"Found {total_products} products matching '{item_name}' across {stores_count} stores. "
        
        if price_analysis and "price_statistics" in price_analysis:
            stats = price_analysis["price_statistics"]
            min_price = stats.get("min_price", 0)
            max_price = stats.get("max_price", 0)
            
            summary += f"Prices range from €{min_price} to €{max_price}. "
            
            if "cheapest_option" in price_analysis:
                cheapest = price_analysis["cheapest_option"]
                summary += f"Cheapest option: {cheapest.get('product_name')} at {cheapest.get('store')} for €{cheapest.get('price')}."
        
        return summary

# Global item finder instance
item_finder = ItemFinder()

async def compare_item_prices(
    item_name: str,
    include_similar: bool = True,
    max_results_per_store: int = 5
) -> Dict[str, Any]:
    """Main function to compare item prices"""
    return await item_finder.compare_item_prices(
        item_name=item_name,
        include_similar=include_similar,
        max_results_per_store=max_results_per_store
    )