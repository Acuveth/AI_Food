#!/usr/bin/env python3
"""
Promotion Finder Module
Finds all promotional items with optional filtering
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from database_handler import get_db_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromotionFinder:
    """
    Handles finding and analyzing promotional items
    """
    
    def __init__(self):
        self.db_handler = None
    
    async def _ensure_db_connection(self):
        """Ensure database connection is available"""
        if self.db_handler is None:
            self.db_handler = await get_db_handler()
    
    async def find_promotions(
        self, 
        search_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        store_filter: Optional[str] = None,
        min_discount: Optional[int] = None,
        max_price: Optional[float] = None,
        sort_by: str = "discount_percentage"
    ) -> Dict[str, Any]:
        """
        Find promotional items with various filters
        
        Args:
            search_filter: Search term to filter product names
            category_filter: Filter by product category
            store_filter: Filter by specific store
            min_discount: Minimum discount percentage
            max_price: Maximum price limit
            sort_by: Sort criteria (discount_percentage, price, name)
        """
        await self._ensure_db_connection()
        
        logger.info(f"🏷️ Finding promotions with filters: search='{search_filter}', category='{category_filter}', store='{store_filter}'")
        
        try:
            # Build dynamic query based on filters
            query = """
            SELECT product_name, store_name, current_price, regular_price, 
                   discount_percentage, ai_main_category, ai_subcategory,
                   ai_health_score, ai_nutrition_grade
            FROM unified_products_view 
            WHERE has_discount = 1 AND current_price > 0
            """
            params = []
            
            # Add filters dynamically
            if search_filter:
                query += " AND product_name LIKE %s"
                params.append(f"%{search_filter}%")
            
            if category_filter:
                query += " AND ai_main_category LIKE %s"
                params.append(f"%{category_filter}%")
            
            if store_filter:
                query += " AND store_name = %s"
                params.append(store_filter.lower())
            
            if min_discount:
                query += " AND discount_percentage >= %s"
                params.append(min_discount)
            
            if max_price:
                query += " AND current_price <= %s"
                params.append(max_price)
            
            # Add sorting
            sort_options = {
                "discount_percentage": "discount_percentage DESC",
                "price": "current_price ASC",
                "name": "product_name ASC",
                "savings": "(regular_price - current_price) DESC"
            }
            
            order_clause = sort_options.get(sort_by, "discount_percentage DESC")
            query += f" ORDER BY {order_clause}, current_price ASC"
            
            # Execute query
            promotions = await self.db_handler.execute_query(query, params)
            
            # Calculate additional metrics
            processed_promotions = self._process_promotion_data(promotions)
            
            # Get analysis insights
            analysis = await self._analyze_promotions(processed_promotions)
            
            # Get category breakdown
            category_breakdown = self._get_category_breakdown(processed_promotions)
            
            # Get store breakdown
            store_breakdown = self._get_store_breakdown(processed_promotions)
            
            result = {
                "success": True,
                "promotions": processed_promotions,
                "total_found": len(processed_promotions),
                "filters_applied": {
                    "search_filter": search_filter,
                    "category_filter": category_filter,
                    "store_filter": store_filter,
                    "min_discount": min_discount,
                    "max_price": max_price,
                    "sort_by": sort_by
                },
                "analysis": analysis,
                "category_breakdown": category_breakdown,
                "store_breakdown": store_breakdown,
                "summary": self._generate_summary(processed_promotions, search_filter)
            }
            
            logger.info(f"✅ Found {len(processed_promotions)} promotional items")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error finding promotions: {e}")
            return {
                "success": False,
                "error": str(e),
                "promotions": [],
                "total_found": 0
            }
    
    def _process_promotion_data(self, promotions: List[Dict]) -> List[Dict]:
        """Process and enhance promotion data with additional calculations"""
        processed = []
        
        for promo in promotions:
            # Calculate savings amount
            regular_price = promo.get('regular_price', 0) or 0
            current_price = promo.get('current_price', 0) or 0
            savings_amount = regular_price - current_price
            
            # Enhance with additional fields
            enhanced_promo = {
                **promo,
                "savings_amount": round(savings_amount, 2),
                "deal_quality": self._assess_deal_quality(promo),
                "value_score": self._calculate_value_score(promo)
            }
            
            processed.append(enhanced_promo)
        
        return processed
    
    def _assess_deal_quality(self, promo: Dict) -> str:
        """Assess the quality of a deal based on discount percentage and other factors"""
        discount = promo.get('discount_percentage', 0) or 0
        health_score = promo.get('ai_health_score', 5) or 5
        
        # Base assessment on discount percentage
        if discount >= 40:
            base_quality = "excellent"
        elif discount >= 25:
            base_quality = "good"
        elif discount >= 15:
            base_quality = "fair"
        else:
            base_quality = "modest"
        
        # Boost quality for healthy items
        if health_score >= 8 and base_quality in ["good", "fair"]:
            base_quality = "excellent"
        elif health_score >= 7 and base_quality == "fair":
            base_quality = "good"
        
        return base_quality
    
    def _calculate_value_score(self, promo: Dict) -> float:
        """Calculate a value score combining discount, price, and health factors"""
        discount = promo.get('discount_percentage', 0) or 0
        current_price = promo.get('current_price', 0) or 0
        health_score = promo.get('ai_health_score', 5) or 5
        
        # Normalize components
        discount_score = min(discount / 50, 1.0)  # Normalize to 0-1
        price_score = max(0, 1 - (current_price / 10))  # Lower price = higher score
        health_score_normalized = health_score / 10
        
        # Weighted combination
        value_score = (discount_score * 0.4 + price_score * 0.3 + health_score_normalized * 0.3)
        
        return round(value_score, 3)
    
    async def _analyze_promotions(self, promotions: List[Dict]) -> Dict[str, Any]:
        """Generate analysis insights for promotions"""
        if not promotions:
            return {"message": "No promotions found to analyze"}
        
        # Basic statistics
        total_promotions = len(promotions)
        avg_discount = sum(p.get('discount_percentage', 0) for p in promotions) / total_promotions
        total_savings = sum(p.get('savings_amount', 0) for p in promotions)
        
        # Best deals
        best_discount = max(promotions, key=lambda x: x.get('discount_percentage', 0))
        best_savings = max(promotions, key=lambda x: x.get('savings_amount', 0))
        best_value = max(promotions, key=lambda x: x.get('value_score', 0))
        
        # Use LLM for deeper analysis if database handler available
        llm_analysis = {}
        if self.db_handler:
            try:
                llm_result = await self.db_handler.analyze_promotions(promotions[:20])
                if llm_result["success"]:
                    llm_analysis = {"llm_insights": llm_result["analysis"]}
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        return {
            "statistics": {
                "total_promotions": total_promotions,
                "average_discount": round(avg_discount, 1),
                "total_potential_savings": round(total_savings, 2),
                "discount_range": {
                    "min": min(p.get('discount_percentage', 0) for p in promotions),
                    "max": max(p.get('discount_percentage', 0) for p in promotions)
                }
            },
            "highlights": {
                "best_discount": {
                    "product": best_discount.get('product_name'),
                    "store": best_discount.get('store_name'),
                    "discount": best_discount.get('discount_percentage'),
                    "price": best_discount.get('current_price')
                },
                "biggest_savings": {
                    "product": best_savings.get('product_name'),
                    "store": best_savings.get('store_name'),
                    "savings": best_savings.get('savings_amount'),
                    "price": best_savings.get('current_price')
                },
                "best_value": {
                    "product": best_value.get('product_name'),
                    "store": best_value.get('store_name'),
                    "value_score": best_value.get('value_score'),
                    "price": best_value.get('current_price')
                }
            },
            **llm_analysis
        }
    
    def _get_category_breakdown(self, promotions: List[Dict]) -> List[Dict]:
        """Get breakdown of promotions by category"""
        category_stats = {}
        
        for promo in promotions:
            category = promo.get('ai_main_category', 'Unknown')
            if category not in category_stats:
                category_stats[category] = {
                    "category": category,
                    "count": 0,
                    "avg_discount": 0,
                    "total_savings": 0,
                    "best_deal": None
                }
            
            stats = category_stats[category]
            stats["count"] += 1
            stats["total_savings"] += promo.get('savings_amount', 0)
            
            # Track best deal in category
            if (stats["best_deal"] is None or 
                promo.get('discount_percentage', 0) > stats["best_deal"].get('discount_percentage', 0)):
                stats["best_deal"] = promo
        
        # Calculate averages
        for stats in category_stats.values():
            if stats["count"] > 0:
                # Calculate average discount for this category
                category_promotions = [p for p in promotions if p.get('ai_main_category') == stats["category"]]
                stats["avg_discount"] = round(
                    sum(p.get('discount_percentage', 0) for p in category_promotions) / len(category_promotions), 1
                )
        
        # Sort by count descending
        return sorted(category_stats.values(), key=lambda x: x["count"], reverse=True)
    
    def _get_store_breakdown(self, promotions: List[Dict]) -> List[Dict]:
        """Get breakdown of promotions by store"""
        store_stats = {}
        
        for promo in promotions:
            store = promo.get('store_name', 'Unknown').upper()
            if store not in store_stats:
                store_stats[store] = {
                    "store": store,
                    "count": 0,
                    "avg_discount": 0,
                    "total_savings": 0,
                    "categories": set()
                }
            
            stats = store_stats[store]
            stats["count"] += 1
            stats["total_savings"] += promo.get('savings_amount', 0)
            if promo.get('ai_main_category'):
                stats["categories"].add(promo.get('ai_main_category'))
        
        # Calculate averages and convert sets to lists
        for stats in store_stats.values():
            if stats["count"] > 0:
                store_promotions = [p for p in promotions if p.get('store_name', '').upper() == stats["store"]]
                stats["avg_discount"] = round(
                    sum(p.get('discount_percentage', 0) for p in store_promotions) / len(store_promotions), 1
                )
            stats["categories"] = list(stats["categories"])
        
        # Sort by count descending
        return sorted(store_stats.values(), key=lambda x: x["count"], reverse=True)
    
    def _generate_summary(self, promotions: List[Dict], search_filter: Optional[str]) -> str:
        """Generate a human-readable summary of the promotion search"""
        if not promotions:
            if search_filter:
                return f"No promotional items found matching '{search_filter}'. Try a broader search term."
            else:
                return "No promotional items are currently available."
        
        total = len(promotions)
        avg_discount = sum(p.get('discount_percentage', 0) for p in promotions) / total
        stores = len(set(p.get('store_name', '') for p in promotions))
        categories = len(set(p.get('ai_main_category', '') for p in promotions))
        
        base_summary = f"Found {total} promotional items"
        
        if search_filter:
            base_summary += f" matching '{search_filter}'"
        
        base_summary += f" across {stores} stores in {categories} categories. "
        base_summary += f"Average discount is {avg_discount:.1f}%."
        
        # Add best deal highlight
        best_deal = max(promotions, key=lambda x: x.get('discount_percentage', 0))
        base_summary += f" Best deal: {best_deal.get('product_name')} at {best_deal.get('store_name', '').upper()} "
        base_summary += f"with {best_deal.get('discount_percentage')}% off."
        
        return base_summary

# Global promotion finder instance
promotion_finder = PromotionFinder()

async def find_promotions(
    search_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    store_filter: Optional[str] = None,
    min_discount: Optional[int] = None,
    max_price: Optional[float] = None,
    sort_by: str = "discount_percentage"
) -> Dict[str, Any]:
    """Main function to find promotions"""
    return await promotion_finder.find_promotions(
        search_filter=search_filter,
        category_filter=category_filter,
        store_filter=store_filter,
        min_discount=min_discount,
        max_price=max_price,
        sort_by=sort_by
    )