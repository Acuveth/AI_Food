#!/usr/bin/env python3
"""
FIXED Product Relevance Evaluator - Handles None values and English/Slovenian translation
Replace your existing product_relevance_evaluator.py with this version
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelevanceType(Enum):
    PRODUCT_NAME_MATCH = "product_name_match"
    CATEGORY_MATCH = "category_match"
    DIETARY_MATCH = "dietary_match"
    PRICE_APPROPRIATENESS = "price_appropriateness"
    STORE_PREFERENCE = "store_preference"
    MEAL_CUISINE_MATCH = "meal_cuisine_match"
    INGREDIENT_ACCURACY = "ingredient_accuracy"
    QUANTITY_APPROPRIATENESS = "quantity_appropriateness"

@dataclass
class UserIntent:
    """Parsed user intent and requirements"""
    original_query: str
    intent_type: str
    target_products: List[str]
    dietary_requirements: List[str]
    cuisine_preferences: List[str]
    price_sensitivity: str
    store_preferences: List[str]
    meal_type: str
    serving_size: Optional[int]
    time_constraint: Optional[int]
    health_focus: bool

@dataclass
class RelevanceScore:
    """Detailed relevance scoring for a single item"""
    item_id: str
    item_name: str
    overall_score: float
    relevance_scores: Dict[RelevanceType, float]
    issues: List[str]
    strengths: List[str]
    explanation: str

@dataclass
class ProductEvaluation:
    """Complete evaluation of product results"""
    user_intent: UserIntent
    total_results: int
    relevant_results: int
    relevance_scores: List[RelevanceScore]
    overall_relevance: float
    category_distribution: Dict[str, int]
    price_distribution: Dict[str, List[float]]
    issues_summary: Dict[str, int]
    recommendations: List[str]

class ProductRelevanceEvaluator:
    """
    FIXED evaluator that handles None values and English/Slovenian translation
    """
    
    def __init__(self):
        # ENGLISH-TO-SLOVENIAN TRANSLATION MAP
        self.translation_map = {
            # Basic products
            "milk": "mleko",
            "cheese": "sir", 
            "bread": "kruh",
            "meat": "meso",
            "chicken": "piščanec",
            "beef": "goveje",
            "pork": "svinjina",
            "fish": "riba",
            "rice": "riž",
            "pasta": "testenine",
            "water": "voda",
            "juice": "sok",
            "coffee": "kava",
            "tea": "čaj",
            "beer": "pivo",
            "wine": "vino",
            "soup": "juha",
            "yogurt": "jogurt",
            "butter": "maslo",
            "cream": "smetana",
            "eggs": "jajca",
            "apple": "jabolko",
            "banana": "banana",
            "orange": "pomaranča",
            "tomato": "paradižnik",
            "potato": "krompir",
            "onion": "čebula",
            "carrot": "korenje",
            # Dietary terms
            "vegetarian": "vegetarijansko",
            "vegan": "veganski",
            "organic": "bio",
            "healthy": "zdravo",
            # Price terms
            "cheap": "poceni",
            "expensive": "drago",
            # Meal types
            "breakfast": "zajtrk",
            "lunch": "kosilo", 
            "dinner": "večerja",
            "snack": "prigrizek"
        }

        # Enhanced product category mappings with both languages
        self.product_categories = {
            "dairy": {
                "primary": ["mleko", "milk", "sir", "cheese", "jogurt", "yogurt", "maslo", "butter", "smetana", "cream"],
                "secondary": ["mlečni", "dairy", "kravji", "cow"]
            },
            "meat": {
                "primary": ["meso", "meat", "piščanec", "chicken", "goveje", "beef", "svinjina", "pork", "jagnjetina", "lamb"],
                "secondary": ["mesni", "meaty"]
            },
            "seafood": {
                "primary": ["riba", "fish", "losos", "salmon", "tuna", "rakovi", "shrimp", "morski", "seafood"],
                "secondary": ["ribji", "fishy"]
            },
            "vegetables": {
                "primary": ["zelenjava", "vegetables", "paradižnik", "tomato", "čebula", "onion", "korenje", "carrot", "krompir", "potato"],
                "secondary": ["zelenjavi", "vegetable"]
            },
            "fruits": {
                "primary": ["sadje", "fruit", "jabolko", "apple", "banana", "pomaranča", "orange", "grozdje", "grape"],
                "secondary": ["sadni", "fruity"]
            },
            "grains": {
                "primary": ["kruh", "bread", "riž", "rice", "testenine", "pasta", "moka", "flour", "žito", "grain", "bageta", "baguette", "toast"],
                "secondary": ["žitni", "grain", "krušni", "bread-like"]
            },
            "beverages": {
                "primary": ["pijača", "beverage", "voda", "water", "sok", "juice", "pivo", "beer", "vino", "wine", "kava", "coffee", "čaj", "tea"],
                "secondary": ["pitni", "drinkable"]
            },
            "soup": {
                "primary": ["juha", "soup", "enolončnica", "stew", "krem", "bisque"],
                "secondary": ["jušni", "soupy"]
            }
        }
        
        # EXCLUSION PATTERNS - Products that should NOT match certain searches
        self.exclusion_patterns = {
            "bread": {
                "exclude_if_contains": ["juha", "soup", "biskvit", "piškot", "torta", "kolač", "kruhki", "drobtine"],
                "exclude_if_primary_category": ["soup", "beverages", "meat", "dairy"]
            },
            "kruh": {  # Slovenian for bread
                "exclude_if_contains": ["juha", "soup", "biskvit", "piškot", "torta", "kolač", "kruhki", "drobtine"],
                "exclude_if_primary_category": ["soup", "beverages", "meat", "dairy"]
            },
            "milk": {
                "exclude_if_contains": ["juha", "soup", "kruhki", "meso", "meat"],
                "exclude_if_primary_category": ["soup", "meat", "grains"]
            },
            "mleko": {  # Slovenian for milk
                "exclude_if_contains": ["juha", "soup", "kruhki", "meso", "meat"],
                "exclude_if_primary_category": ["soup", "meat", "grains"]
            },
            "cheese": {
                "exclude_if_contains": ["juha", "kruhki"],
                "exclude_if_primary_category": ["soup", "grains", "meat"]
            },
            "sir": {  # Slovenian for cheese
                "exclude_if_contains": ["juha", "kruhki"],
                "exclude_if_primary_category": ["soup", "grains", "meat"]
            }
        }
        
        # Dietary requirement keywords
        self.dietary_keywords = {
            "vegetarian": {
                "required": ["vegetarian", "veggie", "plant-based", "vegetarijansko"],
                "forbidden": ["meat", "chicken", "beef", "pork", "fish", "seafood", "meso", "piščanec", "goveje", "svinjina", "riba"]
            },
            "vegan": {
                "required": ["vegan", "plant-based", "veganski"],
                "forbidden": ["meat", "chicken", "beef", "pork", "fish", "dairy", "milk", "cheese", "eggs", "meso", "mleko", "sir", "jajca"]
            },
            "gluten-free": {
                "required": ["gluten-free", "gluten free", "brez glutena"],
                "forbidden": ["wheat", "barley", "rye", "gluten", "pšenica", "ječmen", "gluten"]
            },
            "healthy": {
                "required": ["healthy", "low-fat", "organic", "natural", "zdravo", "bio"],
                "forbidden": ["fried", "processed", "artificial", "ocvrt", "umetno"]
            }
        }
        
        # Price sensitivity mapping
        self.price_ranges = {
            "budget": (0, 3),
            "cheap": (0, 3),
            "poceni": (0, 3),
            "mid-range": (3, 10),
            "premium": (10, float('inf')),
            "expensive": (10, float('inf')),
            "drago": (10, float('inf')),
            "any": (0, float('inf'))
        }

    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely convert value to lowercase string, handling None values"""
        if value is None:
            return default
        return str(value).lower()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, handling None values"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _translate_to_slovenian(self, english_terms: List[str]) -> List[str]:
        """Translate English terms to Slovenian equivalents"""
        translated = []
        for term in english_terms:
            term_lower = term.lower()
            if term_lower in self.translation_map:
                translated.append(self.translation_map[term_lower])
                # Also keep the original English term
                translated.append(term_lower)
            else:
                translated.append(term_lower)
        return translated

    def parse_user_intent(self, query: str, intent_type: str = "unknown") -> UserIntent:
        """Parse user query to extract intent and requirements"""
        query_lower = query.lower()
        
        # Extract target products with translation support
        target_products = []
        
        # Check both English and Slovenian terms
        for category, keywords in self.product_categories.items():
            for keyword in keywords["primary"]:
                if keyword in query_lower:
                    target_products.append(keyword)
        
        # Translate English terms to Slovenian
        if target_products:
            target_products = self._translate_to_slovenian(target_products)
        
        # Extract dietary requirements
        dietary_requirements = []
        for diet, keywords in self.dietary_keywords.items():
            if any(kw in query_lower for kw in keywords["required"]):
                dietary_requirements.append(diet)
        
        # Extract price sensitivity
        price_sensitivity = "any"
        if any(word in query_lower for word in ["cheap", "budget", "affordable", "poceni", "cenovno ugodno"]):
            price_sensitivity = "budget"
        elif any(word in query_lower for word in ["expensive", "premium", "high-end", "drago", "dragoceno"]):
            price_sensitivity = "premium"
        
        # Extract store preferences
        store_preferences = []
        stores = ["dm", "lidl", "mercator", "spar", "tus"]
        for store in stores:
            if store in query_lower:
                store_preferences.append(store)
        
        # Extract meal type
        meal_type = "any"
        if any(word in query_lower for word in ["breakfast", "zajtrk", "morning"]):
            meal_type = "breakfast"
        elif any(word in query_lower for word in ["lunch", "kosilo", "afternoon"]):
            meal_type = "lunch"
        elif any(word in query_lower for word in ["dinner", "večerja", "evening"]):
            meal_type = "dinner"
        
        # Extract serving size and time constraints
        serving_size = None
        serving_matches = re.findall(r'(\d+)\s*(?:people|person|serving|portions?|oseb|oseba)', query_lower)
        if serving_matches:
            serving_size = int(serving_matches[0])
        
        time_constraint = None
        time_matches = re.findall(r'(\d+)\s*(?:min|minute|hour|minut|ura)', query_lower)
        if time_matches:
            time_constraint = int(time_matches[0])
            if any(word in query_lower for word in ["hour", "ura"]):
                time_constraint *= 60
        
        # Health focus
        health_focus = any(word in query_lower for word in ["healthy", "nutritious", "wholesome", "diet", "zdravo", "hranljivo"])
        
        return UserIntent(
            original_query=query,
            intent_type=intent_type,
            target_products=target_products,
            dietary_requirements=dietary_requirements,
            cuisine_preferences=[],
            price_sensitivity=price_sensitivity,
            store_preferences=store_preferences,
            meal_type=meal_type,
            serving_size=serving_size,
            time_constraint=time_constraint,
            health_focus=health_focus
        )
    
    def evaluate_product_relevance(self, product: Dict[str, Any], user_intent: UserIntent) -> RelevanceScore:
        """FIXED evaluation with proper None handling"""
        # SAFE extraction with None handling
        product_name = self._safe_str(product.get("product_name", ""), "unknown product")
        category = self._safe_str(product.get("ai_main_category", ""), "unknown")
        store = self._safe_str(product.get("store_name", ""), "unknown")
        price = self._safe_float(product.get("current_price", 0), 0.0)
        
        relevance_scores = {}
        issues = []
        strengths = []
        
        # CRITICAL: Check exclusion patterns FIRST
        exclusion_score = self._check_exclusion_patterns(product_name, category, user_intent.target_products)
        if exclusion_score < 30:
            logger.info(f"🚫 EXCLUDED: '{product_name}' - Failed exclusion check (score: {exclusion_score})")
            return RelevanceScore(
                item_id=str(hash(product_name)),
                item_name=product.get("product_name", "Unknown Product"),
                overall_score=exclusion_score,
                relevance_scores={},
                issues=[f"Product category mismatch - this is not what user is looking for"],
                strengths=[],
                explanation=f"FILTERED: This product doesn't match the search intent"
            )
        
        # 1. Enhanced Product Name Match
        name_score = self._evaluate_enhanced_name_match(product_name, user_intent.target_products)
        relevance_scores[RelevanceType.PRODUCT_NAME_MATCH] = name_score
        
        if name_score >= 80:
            strengths.append(f"Strong product name match")
        elif name_score < 30:
            issues.append(f"Product name doesn't match search terms")
        
        # 2. Enhanced Category Match
        category_score = self._evaluate_enhanced_category_match(category, product_name, user_intent.target_products)
        relevance_scores[RelevanceType.CATEGORY_MATCH] = category_score
        
        # 3. Dietary Requirements Match
        dietary_score = self._evaluate_dietary_match(product, user_intent.dietary_requirements)
        relevance_scores[RelevanceType.DIETARY_MATCH] = dietary_score
        
        # 4. Price Appropriateness
        price_score = self._evaluate_price_appropriateness(price, user_intent.price_sensitivity)
        relevance_scores[RelevanceType.PRICE_APPROPRIATENESS] = price_score
        
        # 5. Store Preference
        store_score = self._evaluate_store_preference(store, user_intent.store_preferences)
        relevance_scores[RelevanceType.STORE_PREFERENCE] = store_score
        
        # Calculate overall score with weights
        weights = {
            RelevanceType.PRODUCT_NAME_MATCH: 0.4,
            RelevanceType.CATEGORY_MATCH: 0.3,
            RelevanceType.DIETARY_MATCH: 0.2,
            RelevanceType.PRICE_APPROPRIATENESS: 0.08,
            RelevanceType.STORE_PREFERENCE: 0.02
        }
        
        overall_score = sum(score * weights[rel_type] for rel_type, score in relevance_scores.items())
        
        # Apply exclusion penalty
        overall_score = min(overall_score, exclusion_score)
        
        explanation = f"Relevance: {overall_score:.1f}% - " + (strengths[0] if strengths else (issues[0] if issues else "Evaluated"))
        
        return RelevanceScore(
            item_id=str(hash(product_name)),
            item_name=product.get("product_name", "Unknown Product"),
            overall_score=overall_score,
            relevance_scores=relevance_scores,
            issues=issues,
            strengths=strengths,
            explanation=explanation
        )
    
    def _check_exclusion_patterns(self, product_name: str, category: str, target_products: List[str]) -> float:
        """CHECK EXCLUSION PATTERNS with proper None handling"""
        if not target_products:
            return 100
        
        for target_product in target_products:
            target_key = target_product.lower()
            
            # Check if we have exclusion rules for this target
            if target_key not in self.exclusion_patterns:
                continue
            
            exclusion_rules = self.exclusion_patterns[target_key]
            
            # Check if product contains excluded terms
            for excluded_term in exclusion_rules.get("exclude_if_contains", []):
                if excluded_term in product_name:
                    logger.info(f"🚫 EXCLUSION: '{product_name}' contains excluded term '{excluded_term}' for search '{target_product}'")
                    return 15
            
            # Check if product is in excluded category
            excluded_categories = exclusion_rules.get("exclude_if_primary_category", [])
            product_category = self._determine_primary_category(product_name, category)
            if product_category in excluded_categories:
                logger.info(f"🚫 EXCLUSION: '{product_name}' is in excluded category '{product_category}' for search '{target_product}'")
                return 25
        
        return 100
    
    def _determine_primary_category(self, product_name: str, ai_category: str) -> str:
        """Determine the primary category with None handling"""
        if not product_name:
            return ai_category or "unknown"
        
        # Check product name for primary category indicators
        for category, keywords in self.product_categories.items():
            for primary_keyword in keywords["primary"]:
                if primary_keyword in product_name:
                    # Special handling for compound products
                    if "juha" in product_name or "soup" in product_name:
                        return "soup"
                    return category
        
        return ai_category or "unknown"
    
    def _evaluate_enhanced_name_match(self, product_name: str, target_products: List[str]) -> float:
        """Enhanced name matching with None handling"""
        if not target_products or not product_name:
            return 70
        
        max_score = 0
        for target in target_products:
            target_lower = target.lower()
            
            # EXACT MATCH gets highest score
            if target_lower in product_name:
                if product_name.startswith(target_lower) or f" {target_lower} " in f" {product_name} ":
                    score = 100
                else:
                    score = 60
            else:
                # Fuzzy matching
                product_words = set(product_name.split()) if product_name else set()
                target_words = set(target_lower.split())
                overlap = len(product_words.intersection(target_words))
                total_words = len(target_words)
                score = (overlap / total_words) * 70 if total_words > 0 else 0
            
            max_score = max(max_score, score)
        
        return max_score
    
    def _evaluate_enhanced_category_match(self, category: str, product_name: str, target_products: List[str]) -> float:
        """Enhanced category matching with None handling"""
        if not target_products:
            return 70
        
        # Determine what category the user is actually looking for
        user_target_category = None
        for target in target_products:
            for cat, keywords in self.product_categories.items():
                if target.lower() in keywords["primary"]:
                    user_target_category = cat
                    break
        
        if not user_target_category:
            return 50
        
        # Determine actual product category
        product_category = self._determine_primary_category(product_name or "", category or "")
        
        # Direct category match
        if product_category == user_target_category:
            return 95
        
        # Related categories
        related_categories = {
            "grains": ["bakery"],
            "bakery": ["grains"],
            "dairy": ["beverages"],
        }
        
        if user_target_category in related_categories:
            if product_category in related_categories[user_target_category]:
                return 75
        
        return 30
    
    def _evaluate_dietary_match(self, product: Dict[str, Any], dietary_requirements: List[str]) -> float:
        """Evaluate dietary requirements with None handling"""
        if not dietary_requirements:
            return 100
        
        product_name = self._safe_str(product.get("product_name", ""))
        category = self._safe_str(product.get("ai_main_category", ""))
        
        min_score = 100
        for diet in dietary_requirements:
            diet_keywords = self.dietary_keywords.get(diet, {})
            forbidden = diet_keywords.get("forbidden", [])
            
            for forbidden_item in forbidden:
                if forbidden_item in product_name or forbidden_item in category:
                    min_score = min(min_score, 20)
                    break
            else:
                min_score = min(min_score, 85)
        
        return min_score
    
    def _evaluate_price_appropriateness(self, price: float, price_sensitivity: str) -> float:
        """Evaluate price with proper handling"""
        if price <= 0:
            return 50
        
        price_range = self.price_ranges.get(price_sensitivity, (0, float('inf')))
        min_price, max_price = price_range
        
        if min_price <= price <= max_price:
            return 100
        elif price < min_price:
            return 80
        else:
            excess_ratio = (price - max_price) / max_price if max_price > 0 else 1
            return max(20, 80 - (excess_ratio * 60))
    
    def _evaluate_store_preference(self, store: str, store_preferences: List[str]) -> float:
        """Evaluate store preference"""
        if not store_preferences:
            return 100
        
        if store.lower() in [pref.lower() for pref in store_preferences]:
            return 100
        else:
            return 40
    
    def evaluate_meal_relevance(self, meal: Dict[str, Any], user_intent: UserIntent) -> RelevanceScore:
        """Evaluate meal with None handling"""
        meal_title = self._safe_str(meal.get("title", ""), "unknown meal")
        
        return RelevanceScore(
            item_id=str(hash(meal_title)),
            item_name=meal.get("title", "Unknown Meal"),
            overall_score=80.0,  # Default score for meals
            relevance_scores={},
            issues=[],
            strengths=[],
            explanation="Meal evaluation placeholder"
        )
    
    def evaluate_system_output(self, user_query: str, system_response: Dict[str, Any], intent_type: str = "unknown") -> ProductEvaluation:
        """Evaluate system output with proper error handling"""
        try:
            user_intent = self.parse_user_intent(user_query, intent_type)
            
            # Extract items based on response type
            items = []
            if "promotions" in system_response:
                items = system_response["promotions"]
                evaluation_type = "products"
            elif "meals" in system_response:
                items = system_response["meals"]
                evaluation_type = "meals"
            elif "results_by_store" in system_response:
                for store_data in system_response["results_by_store"].values():
                    items.extend(store_data.get("products", []))
                evaluation_type = "products"
            elif "suggested_meals" in system_response:
                items = system_response["suggested_meals"]
                evaluation_type = "meals"
            else:
                items = []
                evaluation_type = "unknown"
            
            # Evaluate each item with error handling
            relevance_scores = []
            for item in items:
                try:
                    if evaluation_type == "meals":
                        score = self.evaluate_meal_relevance(item, user_intent)
                    else:
                        score = self.evaluate_product_relevance(item, user_intent)
                    relevance_scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating item: {e}")
                    # Create a default score for failed evaluations
                    relevance_scores.append(RelevanceScore(
                        item_id="error",
                        item_name="Error evaluating item",
                        overall_score=50.0,
                        relevance_scores={},
                        issues=["Evaluation error"],
                        strengths=[],
                        explanation="Error during evaluation"
                    ))
            
            # Calculate metrics
            total_results = len(items)
            relevant_results = sum(1 for score in relevance_scores if score.overall_score >= 60)
            overall_relevance = sum(score.overall_score for score in relevance_scores) / total_results if total_results > 0 else 0
            
            return ProductEvaluation(
                user_intent=user_intent,
                total_results=total_results,
                relevant_results=relevant_results,
                relevance_scores=relevance_scores,
                overall_relevance=overall_relevance,
                category_distribution={},
                price_distribution={},
                issues_summary={},
                recommendations=[]
            )
            
        except Exception as e:
            logger.error(f"Error in evaluate_system_output: {e}")
            # Return a fallback evaluation
            return ProductEvaluation(
                user_intent=UserIntent(
                    original_query=user_query,
                    intent_type=intent_type,
                    target_products=[],
                    dietary_requirements=[],
                    cuisine_preferences=[],
                    price_sensitivity="any",
                    store_preferences=[],
                    meal_type="any",
                    serving_size=None,
                    time_constraint=None,
                    health_focus=False
                ),
                total_results=0,
                relevant_results=0,
                relevance_scores=[],
                overall_relevance=50.0,
                category_distribution={},
                price_distribution={},
                issues_summary={},
                recommendations=[]
            )