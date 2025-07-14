#!/usr/bin/env python3
"""
Promotion Finder Module - FIXED VERSION
Finds all promotional items with optional filtering
FIXED: Proper None value handling for database fields
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
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling None and invalid values"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=0):
        """Safely convert value to int, handling None and invalid values"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

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
            
            # Calculate additional metrics with safe handling
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
        """Process and enhance promotion data with additional calculations using safe math"""
        processed = []
        
        for promo in promotions:
            # Safe calculation of savings amount
            regular_price = self._safe_float(promo.get('regular_price'))
            current_price = self._safe_float(promo.get('current_price'))
            savings_amount = max(0, regular_price - current_price)  # Ensure non-negative
            
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
        discount = self._safe_float(promo.get('discount_percentage'))
        health_score = self._safe_float(promo.get('ai_health_score'), 5)
        
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
        discount = self._safe_float(promo.get('discount_percentage'))
        current_price = self._safe_float(promo.get('current_price'))
        health_score = self._safe_float(promo.get('ai_health_score'), 5)
        
        # Normalize components with safe division
        discount_score = min(discount / 50, 1.0) if discount > 0 else 0
        price_score = max(0, 1 - (current_price / 10)) if current_price > 0 else 0
        health_score_normalized = health_score / 10 if health_score > 0 else 0.5
        
        # Weighted combination
        value_score = (discount_score * 0.4 + price_score * 0.3 + health_score_normalized * 0.3)
        
        return round(value_score, 3)
    
    async def _analyze_promotions(self, promotions: List[Dict]) -> Dict[str, Any]:
        """Generate analysis insights for promotions with safe calculations"""
        if not promotions:
            return {"message": "No promotions found to analyze"}
        
        # Basic statistics with safe calculations
        total_promotions = len(promotions)
        
        # Safe average discount calculation
        valid_discounts = [self._safe_float(p.get('discount_percentage')) for p in promotions]
        valid_discounts = [d for d in valid_discounts if d > 0]
        avg_discount = sum(valid_discounts) / len(valid_discounts) if valid_discounts else 0
        
        # Safe total savings calculation
        valid_savings = [self._safe_float(p.get('savings_amount')) for p in promotions]
        total_savings = sum(valid_savings)
        
        # Best deals with safe comparisons
        try:
            best_discount = max(promotions, key=lambda x: self._safe_float(x.get('discount_percentage')))
            best_savings = max(promotions, key=lambda x: self._safe_float(x.get('savings_amount')))
            best_value = max(promotions, key=lambda x: self._safe_float(x.get('value_score')))
        except (ValueError, TypeError):
            # Fallback if max operations fail
            best_discount = promotions[0] if promotions else {}
            best_savings = promotions[0] if promotions else {}
            best_value = promotions[0] if promotions else {}
        
        # Use LLM for deeper analysis if database handler available
        llm_analysis = {}
        if self.db_handler:
            try:
                llm_result = await self.db_handler.analyze_promotions(promotions[:20])
                if llm_result["success"]:
                    llm_analysis = {"llm_insights": llm_result["analysis"]}
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # Safe min/max calculations for discount range
        discount_values = [self._safe_float(p.get('discount_percentage')) for p in promotions]
        discount_values = [d for d in discount_values if d > 0]
        
        return {
            "statistics": {
                "total_promotions": total_promotions,
                "average_discount": round(avg_discount, 1),
                "total_potential_savings": round(total_savings, 2),
                "discount_range": {
                    "min": min(discount_values) if discount_values else 0,
                    "max": max(discount_values) if discount_values else 0
                }
            },
            "highlights": {
                "best_discount": {
                    "product": best_discount.get('product_name', 'Unknown'),
                    "store": best_discount.get('store_name', 'Unknown'),
                    "discount": self._safe_float(best_discount.get('discount_percentage')),
                    "price": self._safe_float(best_discount.get('current_price'))
                },
                "biggest_savings": {
                    "product": best_savings.get('product_name', 'Unknown'),
                    "store": best_savings.get('store_name', 'Unknown'),
                    "savings": self._safe_float(best_savings.get('savings_amount')),
                    "price": self._safe_float(best_savings.get('current_price'))
                },
                "best_value": {
                    "product": best_value.get('product_name', 'Unknown'),
                    "store": best_value.get('store_name', 'Unknown'),
                    "value_score": self._safe_float(best_value.get('value_score')),
                    "price": self._safe_float(best_value.get('current_price'))
                }
            },
            **llm_analysis
        }
    
    def _get_category_breakdown(self, promotions: List[Dict]) -> List[Dict]:
        """Get breakdown of promotions by category with safe calculations"""
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
            stats["total_savings"] += self._safe_float(promo.get('savings_amount'))
            
            # Track best deal in category
            current_discount = self._safe_float(promo.get('discount_percentage'))
            best_discount = self._safe_float(stats["best_deal"].get('discount_percentage') if stats["best_deal"] else 0)
            
            if stats["best_deal"] is None or current_discount > best_discount:
                stats["best_deal"] = promo
        
        # Calculate averages with safe math
        for stats in category_stats.values():
            if stats["count"] > 0:
                # Calculate average discount for this category
                category_promotions = [p for p in promotions if p.get('ai_main_category') == stats["category"]]
                valid_discounts = [self._safe_float(p.get('discount_percentage')) for p in category_promotions]
                valid_discounts = [d for d in valid_discounts if d > 0]
                
                if valid_discounts:
                    stats["avg_discount"] = round(sum(valid_discounts) / len(valid_discounts), 1)
                else:
                    stats["avg_discount"] = 0
        
        # Sort by count descending
        return sorted(category_stats.values(), key=lambda x: x["count"], reverse=True)
    
    def _get_store_breakdown(self, promotions: List[Dict]) -> List[Dict]:
        """Get breakdown of promotions by store with safe calculations"""
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
            stats["total_savings"] += self._safe_float(promo.get('savings_amount'))
            if promo.get('ai_main_category'):
                stats["categories"].add(promo.get('ai_main_category'))
        
        # Calculate averages and convert sets to lists with safe math
        for stats in store_stats.values():
            if stats["count"] > 0:
                store_promotions = [p for p in promotions if p.get('store_name', '').upper() == stats["store"]]
                valid_discounts = [self._safe_float(p.get('discount_percentage')) for p in store_promotions]
                valid_discounts = [d for d in valid_discounts if d > 0]
                
                if valid_discounts:
                    stats["avg_discount"] = round(sum(valid_discounts) / len(valid_discounts), 1)
                else:
                    stats["avg_discount"] = 0
            
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
        
        # Safe calculation of average discount
        valid_discounts = [self._safe_float(p.get('discount_percentage')) for p in promotions]
        valid_discounts = [d for d in valid_discounts if d > 0]
        avg_discount = sum(valid_discounts) / len(valid_discounts) if valid_discounts else 0
        
        stores = len(set(p.get('store_name', '') for p in promotions))
        categories = len(set(p.get('ai_main_category', '') for p in promotions))
        
        base_summary = f"Found {total} promotional items"
        
        if search_filter:
            base_summary += f" matching '{search_filter}'"
        
        base_summary += f" across {stores} stores in {categories} categories. "
        base_summary += f"Average discount is {avg_discount:.1f}%."
        
        # Add best deal highlight with safe access
        try:
            best_deal = max(promotions, key=lambda x: self._safe_float(x.get('discount_percentage')))
            discount_pct = self._safe_float(best_deal.get('discount_percentage'))
            product_name = best_deal.get('product_name', 'Unknown product')
            store_name = best_deal.get('store_name', 'Unknown store').upper()
            
            base_summary += f" Best deal: {product_name} at {store_name} "
            base_summary += f"with {discount_pct:.0f}% off."
        except (ValueError, TypeError):
            base_summary += " Check individual items for best deals."
        
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