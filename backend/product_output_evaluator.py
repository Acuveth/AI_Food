#!/usr/bin/env python3
"""
ENHANCED Product Relevance Evaluator - Comprehensive Slovenian Language Support
Handles None values and provides extensive English/Slovenian translation and mapping
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
    ENHANCED evaluator with comprehensive Slovenian language support
    """
    
    def __init__(self):
        # COMPREHENSIVE ENGLISH-TO-SLOVENIAN TRANSLATION MAP
        self.translation_map = {
            # Basic food products
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
            "sugar": "sladkor",
            "salt": "sol",
            "flour": "moka",
            "oil": "olje",
            "vinegar": "kis",
            
            # Fruits
            "apple": "jabolko",
            "banana": "banana",
            "orange": "pomaranča",
            "grape": "grozdje",
            "strawberry": "jagoda",
            "pear": "hruška",
            "peach": "breskev",
            "plum": "sliva",
            "cherry": "češnja",
            "lemon": "limona",
            
            # Vegetables
            "tomato": "paradižnik",
            "potato": "krompir",
            "onion": "čebula",
            "carrot": "korenje",
            "cabbage": "zelje",
            "lettuce": "solata",
            "cucumber": "kumara",
            "pepper": "paprika",
            "garlic": "česen",
            "spinach": "špinača",
            
            # Beverages
            "mineral water": "mineralna voda",
            "sparkling water": "gazirana voda",
            "fruit juice": "sadni sok",
            "orange juice": "pomarančni sok",
            "apple juice": "jabolčni sok",
            "soft drink": "brezalkoholna pijača",
            "soda": "gazirana pijača",
            
            # Dairy products
            "whole milk": "polnomastno mleko",
            "skim milk": "posneto mleko",
            "cottage cheese": "skuta",
            "cream cheese": "kremni sir",
            "hard cheese": "trdi sir",
            "soft cheese": "mehki sir",
            "fresh cheese": "sveži sir",
            
            # Meat products
            "chicken breast": "piščančji file",
            "chicken thigh": "piščančji stegen",
            "ground beef": "goveja sekanica",
            "pork chop": "svinjski kotlet",
            "ham": "šunka",
            "bacon": "slanina",
            "sausage": "klobasa",
            "salami": "salama",
            
            # Grains and cereals
            "white bread": "bel kruh",
            "brown bread": "črn kruh",
            "whole grain bread": "polnozrnati kruh",
            "toast": "toast",
            "roll": "žemlja",
            "pasta": "testenine",
            "spaghetti": "špageti",
            "macaroni": "makaroni",
            "white rice": "bel riž",
            "brown rice": "rjav riž",
            "oats": "ovseni kosmiči",
            "cereals": "žitarice",
            
            # Processed foods
            "cookies": "piškoti",
            "crackers": "krekerji",
            "chips": "čips",
            "chocolate": "čokolada",
            "candy": "bonboni",
            "ice cream": "sladoled",
            "cake": "torta",
            "pastry": "pecivo",
            
            # Dietary terms
            "vegetarian": "vegetarijansko",
            "vegan": "veganski",
            "organic": "bio",
            "natural": "naravno",
            "healthy": "zdravo",
            "low fat": "z malo maščobe",
            "gluten free": "brez glutena",
            "sugar free": "brez sladkorja",
            "lactose free": "brez laktoze",
            
            # Price terms
            "cheap": "poceni",
            "expensive": "drago",
            "discount": "popust",
            "sale": "akcija",
            "promotion": "ponudba",
            "deal": "ugodna ponudba",
            "bargain": "ugoden nakup",
            
            # Meal types
            "breakfast": "zajtrk",
            "lunch": "kosilo", 
            "dinner": "večerja",
            "snack": "prigrizek",
            "dessert": "sladica",
            "appetizer": "predjed",
            
            # Cooking terms
            "fresh": "sveže",
            "frozen": "zamrznjeno",
            "canned": "konzervirano",
            "dried": "sušeno",
            "smoked": "dimljeno",
            "grilled": "žarjeno",
            "fried": "cvrt",
            "baked": "pečeno",
            "cooked": "kuhano",
            
            # Quantities
            "liter": "liter",
            "kilogram": "kilogram",
            "gram": "gram",
            "piece": "kos",
            "package": "paket",
            "bottle": "steklenica",
            "can": "konzerva",
            "box": "škatla"
        }

        # REVERSE TRANSLATION MAP (Slovenian to English)
        self.reverse_translation_map = {v: k for k, v in self.translation_map.items()}
        
        # Add additional Slovenian variations
        self.reverse_translation_map.update({
            "mlečni izdelki": "dairy products",
            "mesni izdelki": "meat products",
            "pekovski izdelki": "bakery products",
            "sadje": "fruit",
            "zelenjava": "vegetables",
            "pijače": "beverages",
            "alkoholne pijače": "alcoholic beverages",
            "brezalkoholne pijače": "non-alcoholic beverages",
            "sladkarije": "sweets",
            "zamrznjeni izdelki": "frozen products",
            "konzervirani izdelki": "canned products",
            "organski izdelki": "organic products",
            "bio izdelki": "organic products",
            "dietetski izdelki": "dietary products",
            "začimbe": "spices",
            "pripomočki": "accessories",
            "čistila": "cleaning products",
            "higiena": "hygiene products",
            "otroška hrana": "baby food",
            "hrana za živali": "pet food"
        })

        # Enhanced product category mappings with comprehensive Slovenian + English terms
        self.product_categories = {
            "dairy": {
                "slovenian": ["mleko", "sir", "jogurt", "maslo", "smetana", "skuta", "mlečni izdelki", "kefir", "kislo mleko"],
                "english": ["milk", "cheese", "yogurt", "butter", "cream", "cottage cheese", "dairy", "kefir", "sour cream"],
                "brands": ["Ljubljanske mlekarne", "Alpsko mleko", "Mu", "Planica", "Alpro"]
            },
            "meat": {
                "slovenian": ["meso", "piščanec", "goveje", "svinjina", "mesni izdelki", "šunka", "klobasa", "slanina", "perutnina"],
                "english": ["meat", "chicken", "beef", "pork", "meat products", "ham", "sausage", "bacon", "poultry"],
                "brands": ["Pik", "Carnex", "Kosaki", "Perutnina Ptuj", "Kmetija Jevšček"]
            },
            "seafood": {
                "slovenian": ["riba", "morski sadeži", "losos", "tuna", "postrv", "rake", "školjke", "ribje izdelki"],
                "english": ["fish", "seafood", "salmon", "tuna", "trout", "crab", "shellfish", "fish products"],
                "brands": ["Fangst", "Tuna", "Delamaris", "Droga Kolinska"]
            },
            "vegetables": {
                "slovenian": ["zelenjava", "paradižnik", "čebula", "krompir", "korenje", "zelje", "solata", "kumara", "paprika"],
                "english": ["vegetables", "tomato", "onion", "potato", "carrot", "cabbage", "lettuce", "cucumber", "pepper"],
                "brands": ["Dolenjski", "Natura", "Bio zelenjava", "Kmetija Skok"]
            },
            "fruits": {
                "slovenian": ["sadje", "jabolka", "banane", "pomaranče", "grozdje", "jagode", "breskve", "sadni izdelki"],
                "english": ["fruit", "apples", "bananas", "oranges", "grapes", "strawberries", "peaches", "fruit products"],
                "brands": ["Fructal", "Natura", "Bio sadje", "Kmetija Čelik"]
            },
            "grains": {
                "slovenian": ["kruh", "testenine", "riž", "žitarice", "pekovski izdelki", "moka", "ovseni kosmiči", "pecivo"],
                "english": ["bread", "pasta", "rice", "cereals", "bakery products", "flour", "oats", "pastry"],
                "brands": ["Žito", "Penam", "Barilla", "Zlato polje", "Mercator"]
            },
            "beverages": {
                "slovenian": ["pijače", "voda", "sok", "kava", "čaj", "mineralna voda", "gazirana pijača", "sadni sok"],
                "english": ["beverages", "water", "juice", "coffee", "tea", "mineral water", "soft drink", "fruit juice"],
                "brands": ["Radenska", "Fructal", "Cockta", "Pivka", "Tetra pak", "Jamnica"]
            },
            "snacks": {
                "slovenian": ["prigrizki", "čips", "krekerji", "oreški", "piškoti", "sladkarije", "čokolada"],
                "english": ["snacks", "chips", "crackers", "nuts", "cookies", "sweets", "chocolate"],
                "brands": ["Droga Kolinska", "Štark", "Kraš", "Gorenjka", "Kras"]
            },
            "frozen": {
                "slovenian": ["zamrznjeni izdelki", "zamrznjena zelenjava", "zamrznjeno sadje", "sladoled", "zamrznjene jedi"],
                "english": ["frozen products", "frozen vegetables", "frozen fruit", "ice cream", "frozen meals"],
                "brands": ["Frigo", "Ledo", "Adria", "Zala", "Mercator"]
            },
            "alcoholic": {
                "slovenian": ["alkoholne pijače", "pivo", "vino", "žganje", "aperitiv", "whiskey", "vodka"],
                "english": ["alcoholic beverages", "beer", "wine", "spirits", "aperitif", "whiskey", "vodka"],
                "brands": ["Laško", "Union", "Radgonske gorice", "Vipava", "Cviček"]
            }
        }
        
        # COMPREHENSIVE EXCLUSION PATTERNS for Slovenian products
        self.exclusion_patterns = {
            # Slovenian terms
            "mleko": {
                "exclude_if_contains": ["juha", "kruh", "meso", "klobasa", "čistilo", "pralno"],
                "exclude_if_primary_category": ["meat", "grains", "cleaning", "hygiene"]
            },
            "kruh": {
                "exclude_if_contains": ["mleko", "jogurt", "meso", "riba", "juha", "čistilo"],
                "exclude_if_primary_category": ["dairy", "meat", "beverages", "cleaning"]
            },
            "sir": {
                "exclude_if_contains": ["kruh", "meso", "juha", "čistilo", "pralno"],
                "exclude_if_primary_category": ["grains", "meat", "cleaning", "hygiene"]
            },
            "meso": {
                "exclude_if_contains": ["mleko", "kruh", "sadje", "zelenjava", "čistilo"],
                "exclude_if_primary_category": ["dairy", "grains", "fruits", "vegetables", "cleaning"]
            },
            "piščanec": {
                "exclude_if_contains": ["mleko", "kruh", "sadje", "čistilo", "hrana za živali"],
                "exclude_if_primary_category": ["dairy", "grains", "fruits", "cleaning", "pet_food"]
            },
            "riba": {
                "exclude_if_contains": ["mleko", "kruh", "sadje", "zelenjava", "čistilo"],
                "exclude_if_primary_category": ["dairy", "grains", "fruits", "vegetables", "cleaning"]
            },
            "voda": {
                "exclude_if_contains": ["meso", "kruh", "sir", "čistilo", "tehnično"],
                "exclude_if_primary_category": ["meat", "grains", "dairy", "cleaning", "technical"]
            },
            "sok": {
                "exclude_if_contains": ["meso", "kruh", "sir", "čistilo", "alkohol"],
                "exclude_if_primary_category": ["meat", "grains", "dairy", "cleaning", "alcoholic"]
            },
            # English terms
            "milk": {
                "exclude_if_contains": ["juha", "soup", "kruh", "bread", "meat", "meso", "cleaning", "čistilo"],
                "exclude_if_primary_category": ["soup", "grains", "meat", "cleaning"]
            },
            "bread": {
                "exclude_if_contains": ["mleko", "milk", "jogurt", "yogurt", "meso", "meat", "čistilo"],
                "exclude_if_primary_category": ["dairy", "meat", "beverages", "cleaning"]
            },
            "cheese": {
                "exclude_if_contains": ["kruh", "bread", "meso", "meat", "čistilo", "cleaning"],
                "exclude_if_primary_category": ["grains", "meat", "cleaning"]
            },
            "chicken": {
                "exclude_if_contains": ["mleko", "milk", "kruh", "bread", "čistilo", "pet food"],
                "exclude_if_primary_category": ["dairy", "grains", "cleaning", "pet_food"]
            },
            "water": {
                "exclude_if_contains": ["meso", "meat", "kruh", "bread", "čistilo", "cleaning"],
                "exclude_if_primary_category": ["meat", "grains", "dairy", "cleaning"]
            }
        }
        
        # Enhanced dietary requirement keywords with Slovenian support
        self.dietary_keywords = {
            "vegetarian": {
                "slovenian": ["vegetarijansko", "vegetarijski", "brez mesa", "rastlinsko"],
                "english": ["vegetarian", "veggie", "plant-based", "meat-free"],
                "forbidden": [
                    # English
                    "meat", "chicken", "beef", "pork", "fish", "seafood", 
                    # Slovenian
                    "meso", "piščanec", "goveje", "svinjina", "riba", "morski sadeži",
                    "šunka", "klobasa", "slanina", "perutnina"
                ]
            },
            "vegan": {
                "slovenian": ["veganski", "veganska", "brez živalskih", "rastlinsko"],
                "english": ["vegan", "plant-based", "dairy-free", "animal-free"],
                "forbidden": [
                    # English
                    "meat", "chicken", "beef", "pork", "fish", "dairy", "milk", 
                    "cheese", "eggs", "butter", "cream", "yogurt", "honey",
                    # Slovenian
                    "meso", "piščanec", "goveje", "svinjina", "riba", "mleko", 
                    "sir", "jajca", "maslo", "smetana", "jogurt", "med",
                    "mlečni izdelki", "živalski izdelki"
                ]
            },
            "gluten-free": {
                "slovenian": ["brez glutena", "bezglutenski", "brez pšenice"],
                "english": ["gluten-free", "gluten free", "wheat-free"],
                "forbidden": [
                    # English
                    "gluten", "wheat", "barley", "rye", "flour", "bread", "pasta",
                    # Slovenian
                    "gluten", "pšenica", "ječmen", "rž", "moka", "kruh", "testenine"
                ]
            },
            "organic": {
                "slovenian": ["bio", "organski", "naravno", "ekološki"],
                "english": ["organic", "bio", "natural", "ecological"],
                "forbidden": [
                    # English
                    "artificial", "synthetic", "chemical", "processed",
                    # Slovenian
                    "umetno", "sintetično", "kemično", "procesiran"
                ]
            },
            "healthy": {
                "slovenian": ["zdravo", "zdrav", "nizka maščoba", "malo kalorij"],
                "english": ["healthy", "low-fat", "low-calorie", "nutritious"],
                "forbidden": [
                    # English
                    "fried", "processed", "high-fat", "junk",
                    # Slovenian
                    "cvrt", "procesiran", "visoka maščoba", "nezdravo"
                ]
            }
        }
        
        # Enhanced price sensitivity mapping with Slovenian terms
        self.price_ranges = {
            # English terms
            "budget": (0, 3),
            "cheap": (0, 3),
            "mid-range": (3, 10),
            "premium": (10, float('inf')),
            "expensive": (10, float('inf')),
            "any": (0, float('inf')),
            # Slovenian terms
            "poceni": (0, 3),
            "ugodno": (0, 3),
            "cenovno ugodno": (0, 3),
            "srednje": (3, 10),
            "običajno": (3, 10),
            "drago": (10, float('inf')),
            "premium": (10, float('inf')),
            "luksuzno": (15, float('inf')),
            "karkoli": (0, float('inf')),
            "vseeno": (0, float('inf'))
        }

        # Store name mappings
        self.store_mappings = {
            "tuš": "tus",
            "tus": "tus",
            "dm": "dm",
            "lidl": "lidl",
            "mercator": "mercator",
            "spar": "spar",
            "hofer": "hofer",  # Austrian discount chain
            "interspar": "interspar",
            "leclerc": "leclerc"
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
        return translated

    def _translate_to_english(self, slovenian_terms: List[str]) -> List[str]:
        """Translate Slovenian terms to English equivalents"""
        translated = []
        for term in slovenian_terms:
            term_lower = term.lower()
            if term_lower in self.reverse_translation_map:
                translated.append(self.reverse_translation_map[term_lower])
            # Also keep the original Slovenian term
            translated.append(term_lower)
        return translated

    def _get_product_synonyms(self, product_name: str) -> List[str]:
        """Get synonyms for a product in both languages"""
        synonyms = [product_name.lower()]
        
        # Check if it's an English term
        if product_name.lower() in self.translation_map:
            synonyms.append(self.translation_map[product_name.lower()])
        
        # Check if it's a Slovenian term
        if product_name.lower() in self.reverse_translation_map:
            synonyms.append(self.reverse_translation_map[product_name.lower()])
        
        # Add category-specific synonyms
        for category, data in self.product_categories.items():
            if product_name.lower() in data["slovenian"] or product_name.lower() in data["english"]:
                synonyms.extend(data["slovenian"])
                synonyms.extend(data["english"])
                break
        
        return list(set(synonyms))

    def parse_user_intent(self, query: str, intent_type: str = "unknown") -> UserIntent:
        """Parse user query to extract intent and requirements with comprehensive Slovenian support"""
        query_lower = query.lower()
        
        # Extract target products with comprehensive translation support
        target_products = []
        
        # Check for direct mentions of products
        for english_term, slovenian_term in self.translation_map.items():
            if english_term in query_lower:
                target_products.extend([english_term, slovenian_term])
            if slovenian_term in query_lower:
                target_products.extend([slovenian_term, english_term])
        
        # Check category-specific terms
        for category, data in self.product_categories.items():
            for term in data["slovenian"] + data["english"]:
                if term in query_lower:
                    target_products.append(term)
                    # Add synonyms from the same category
                    target_products.extend(data["slovenian"][:3])  # Add first 3 Slovenian terms
                    target_products.extend(data["english"][:3])   # Add first 3 English terms
                    break
        
        # Extract dietary requirements with Slovenian support
        dietary_requirements = []
        for diet_key, diet_data in self.dietary_keywords.items():
            slovenian_terms = diet_data.get("slovenian", [])
            english_terms = diet_data.get("english", [])
            
            if any(term in query_lower for term in slovenian_terms + english_terms):
                dietary_requirements.append(diet_key)
        
        # Extract price sensitivity with Slovenian support
        price_sensitivity = "any"
        for price_term, price_range in self.price_ranges.items():
            if price_term in query_lower:
                price_sensitivity = price_term
                break
        
        # Extract store preferences with Slovenian support
        store_preferences = []
        for store_variant, canonical_name in self.store_mappings.items():
            if store_variant in query_lower:
                store_preferences.append(canonical_name)
        
        # Extract meal type with Slovenian support
        meal_type = "any"
        slovenian_meals = {
            "zajtrk": "breakfast",
            "kosilo": "lunch",
            "večerja": "dinner",
            "malica": "snack",
            "sladica": "dessert"
        }
        
        for slo_meal, eng_meal in slovenian_meals.items():
            if slo_meal in query_lower or eng_meal in query_lower:
                meal_type = eng_meal
                break
        
        # Extract serving size and time constraints
        serving_size = None
        serving_matches = re.findall(r'(\d+)\s*(?:people|person|serving|portions?|oseb|oseba|porcij|porcije)', query_lower)
        if serving_matches:
            serving_size = int(serving_matches[0])
        
        time_constraint = None
        time_matches = re.findall(r'(\d+)\s*(?:min|minute|hour|minut|ura|ur)', query_lower)
        if time_matches:
            time_constraint = int(time_matches[0])
            if any(word in query_lower for word in ["hour", "ura", "ur"]):
                time_constraint *= 60
        
        # Health focus with Slovenian support
        health_focus = any(word in query_lower for word in [
            "healthy", "nutritious", "wholesome", "diet", "fitness",
            "zdravo", "zdravi", "hranljivo", "dieta", "fitnes"
        ])
        
        # Remove duplicates from target products
        target_products = list(set(target_products))
        
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
        """ENHANCED evaluation with comprehensive Slovenian support"""
        # SAFE extraction with None handling
        product_name = self._safe_str(product.get("product_name", ""), "unknown product")
        category = self._safe_str(product.get("ai_main_category", ""), "unknown")
        store = self._safe_str(product.get("store_name", ""), "unknown")
        price = self._safe_float(product.get("current_price", 0), 0.0)
        
        relevance_scores = {}
        issues = []
        strengths = []
        
        # CRITICAL: Check exclusion patterns FIRST with Slovenian support
        exclusion_score = self._check_exclusion_patterns_enhanced(product_name, category, user_intent.target_products)
        if exclusion_score < 30:
            logger.info(f"🚫 EXCLUDED: '{product_name}' - Failed exclusion check (score: {exclusion_score})")
            return RelevanceScore(
                item_id=str(hash(product_name)),
                item_name=product.get("product_name", "Unknown Product"),
                overall_score=exclusion_score,
                relevance_scores={},
                issues=[f"Kategoria izdelka se ne ujema - to ni tisto, kar uporabnik išče"],
                strengths=[],
                explanation=f"FILTRIRANO: Ta izdelek ne ustreza iskalnemu namenu"
            )
        
        # 1. Enhanced Product Name Match with Slovenian support
        name_score = self._evaluate_enhanced_name_match_slovenian(product_name, user_intent.target_products)
        relevance_scores[RelevanceType.PRODUCT_NAME_MATCH] = name_score
        
        if name_score >= 80:
            strengths.append(f"Močno ujemanje imena izdelka")
        elif name_score < 30:
            issues.append(f"Ime izdelka se ne ujema z iskalnimi pojmi")
        
        # 2. Enhanced Category Match with Slovenian support
        category_score = self._evaluate_enhanced_category_match_slovenian(category, product_name, user_intent.target_products)
        relevance_scores[RelevanceType.CATEGORY_MATCH] = category_score
        
        # 3. Dietary Requirements Match with Slovenian support
        dietary_score = self._evaluate_dietary_match_slovenian(product, user_intent.dietary_requirements)
        relevance_scores[RelevanceType.DIETARY_MATCH] = dietary_score
        
        # 4. Price Appropriateness with Slovenian support
        price_score = self._evaluate_price_appropriateness_slovenian(price, user_intent.price_sensitivity)
        relevance_scores[RelevanceType.PRICE_APPROPRIATENESS] = price_score
        
        # 5. Store Preference with Slovenian support
        store_score = self._evaluate_store_preference_slovenian(store, user_intent.store_preferences)
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
        
        # Generate explanation in Slovenian
        explanation = f"Relevantnost: {overall_score:.1f}% - " + (strengths[0] if strengths else (issues[0] if issues else "Ocenjeno"))
        
        return RelevanceScore(
            item_id=str(hash(product_name)),
            item_name=product.get("product_name", "Unknown Product"),
            overall_score=overall_score,
            relevance_scores=relevance_scores,
            issues=issues,
            strengths=strengths,
            explanation=explanation
        )
    
    def _check_exclusion_patterns_enhanced(self, product_name: str, category: str, target_products: List[str]) -> float:
        """Enhanced exclusion pattern checking with Slovenian support"""
        if not target_products:
            return 100
        
        for target_product in target_products:
            target_key = target_product.lower()
            
            # Check both direct key and translated versions
            check_keys = [target_key]
            if target_key in self.translation_map:
                check_keys.append(self.translation_map[target_key])
            if target_key in self.reverse_translation_map:
                check_keys.append(self.reverse_translation_map[target_key])
            
            for check_key in check_keys:
                if check_key in self.exclusion_patterns:
                    exclusion_rules = self.exclusion_patterns[check_key]
                    
                    # Check if product contains excluded terms
                    for excluded_term in exclusion_rules.get("exclude_if_contains", []):
                        if excluded_term in product_name:
                            logger.info(f"🚫 EXCLUSION: '{product_name}' contains excluded term '{excluded_term}' for search '{target_product}'")
                            return 15
                    
                    # Check if product is in excluded category
                    excluded_categories = exclusion_rules.get("exclude_if_primary_category", [])
                    product_category = self._determine_primary_category_enhanced(product_name, category)
                    if product_category in excluded_categories:
                        logger.info(f"🚫 EXCLUSION: '{product_name}' is in excluded category '{product_category}' for search '{target_product}'")
                        return 25
        
        return 100
    
    def _determine_primary_category_enhanced(self, product_name: str, ai_category: str) -> str:
        """Enhanced primary category determination with Slovenian support"""
        if not product_name:
            return ai_category or "unknown"
        
        # Check product name against comprehensive category mappings
        for category, data in self.product_categories.items():
            slovenian_terms = data["slovenian"]
            english_terms = data["english"]
            
            if any(term in product_name for term in slovenian_terms + english_terms):
                # Special handling for compound products
                if any(term in product_name for term in ["juha", "soup"]):
                    return "soup"
                return category
        
        return ai_category or "unknown"
    
    def _evaluate_enhanced_name_match_slovenian(self, product_name: str, target_products: List[str]) -> float:
        """Enhanced name matching with comprehensive Slovenian support"""
        if not target_products or not product_name:
            return 70
        
        max_score = 0
        for target in target_products:
            target_lower = target.lower()
            
            # Get synonyms for the target term
            synonyms = self._get_product_synonyms(target_lower)
            
            # Check for exact matches with synonyms
            for synonym in synonyms:
                if synonym in product_name:
                    if product_name.startswith(synonym) or f" {synonym} " in f" {product_name} ":
                        score = 100
                    else:
                        score = 85
                    max_score = max(max_score, score)
            
            # Fuzzy matching with word overlap
            product_words = set(product_name.split()) if product_name else set()
            target_words = set(target_lower.split())
            
            # Also check synonym words
            for synonym in synonyms:
                target_words.update(synonym.split())
            
            overlap = len(product_words.intersection(target_words))
            total_words = len(target_words)
            if total_words > 0:
                fuzzy_score = (overlap / total_words) * 75
                max_score = max(max_score, fuzzy_score)
        
        return max_score
    
    def _evaluate_enhanced_category_match_slovenian(self, category: str, product_name: str, target_products: List[str]) -> float:
        """Enhanced category matching with comprehensive Slovenian support"""
        if not target_products:
            return 70
        
        # Determine what category the user is actually looking for
        user_target_category = None
        for target in target_products:
            target_lower = target.lower()
            
            # Check in comprehensive category mappings
            for cat, data in self.product_categories.items():
                slovenian_terms = data["slovenian"]
                english_terms = data["english"]
                
                if target_lower in slovenian_terms or target_lower in english_terms:
                    user_target_category = cat
                    break
            
            if user_target_category:
                break
        
        if not user_target_category:
            return 50
        
        # Determine actual product category
        product_category = self._determine_primary_category_enhanced(product_name or "", category or "")
        
        # Direct category match
        if product_category == user_target_category:
            return 95
        
        # Related categories
        related_categories = {
            "grains": ["snacks", "frozen"],
            "dairy": ["beverages", "frozen"],
            "meat": ["frozen", "snacks"],
            "vegetables": ["frozen", "snacks"],
            "fruits": ["beverages", "frozen"],
            "beverages": ["dairy", "snacks"]
        }
        
        if user_target_category in related_categories:
            if product_category in related_categories[user_target_category]:
                return 75
        
        return 30
    
    def _evaluate_dietary_match_slovenian(self, product: Dict[str, Any], dietary_requirements: List[str]) -> float:
        """Enhanced dietary requirements evaluation with Slovenian support"""
        if not dietary_requirements:
            return 100
        
        product_name = self._safe_str(product.get("product_name", ""))
        category = self._safe_str(product.get("ai_main_category", ""))
        
        min_score = 100
        for diet in dietary_requirements:
            diet_lower = diet.lower()
            
            # Normalize diet names
            if diet_lower in ["vegetarijansko", "vegetarijski"]:
                diet_lower = "vegetarian"
            elif diet_lower in ["veganski", "veganska"]:
                diet_lower = "vegan"
            elif diet_lower in ["brez glutena", "bezglutenski"]:
                diet_lower = "gluten-free"
            elif diet_lower in ["bio", "organski"]:
                diet_lower = "organic"
            elif diet_lower in ["zdravo", "zdrav"]:
                diet_lower = "healthy"
            
            if diet_lower in self.dietary_keywords:
                diet_data = self.dietary_keywords[diet_lower]
                forbidden = diet_data.get("forbidden", [])
                
                for forbidden_item in forbidden:
                    if forbidden_item in product_name or forbidden_item in category:
                        min_score = min(min_score, 20)
                        break
                else:
                    # Check if product explicitly mentions the diet
                    slovenian_terms = diet_data.get("slovenian", [])
                    english_terms = diet_data.get("english", [])
                    
                    if any(term in product_name for term in slovenian_terms + english_terms):
                        min_score = min(min_score, 100)
                    else:
                        min_score = min(min_score, 85)
        
        return min_score
    
    def _evaluate_price_appropriateness_slovenian(self, price: float, price_sensitivity: str) -> float:
        """Enhanced price evaluation with Slovenian support"""
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
    
    def _evaluate_store_preference_slovenian(self, store: str, store_preferences: List[str]) -> float:
        """Enhanced store preference evaluation with Slovenian support"""
        if not store_preferences:
            return 100
        
        # Normalize store name
        store_normalized = self.store_mappings.get(store.lower(), store.lower())
        
        # Normalize preferences
        preferences_normalized = []
        for pref in store_preferences:
            preferences_normalized.append(self.store_mappings.get(pref.lower(), pref.lower()))
        
        if store_normalized in preferences_normalized:
            return 100
        else:
            return 40
    
    def evaluate_meal_relevance(self, meal: Dict[str, Any], user_intent: UserIntent) -> RelevanceScore:
        """Enhanced meal evaluation with Slovenian support"""
        meal_title = self._safe_str(meal.get("title", ""), "unknown meal")
        
        return RelevanceScore(
            item_id=str(hash(meal_title)),
            item_name=meal.get("title", "Unknown Meal"),
            overall_score=80.0,  # Default score for meals
            relevance_scores={},
            issues=[],
            strengths=[],
            explanation="Ocena jedi z osnovno podporo"
        )
    
    def evaluate_system_output(self, user_query: str, system_response: Dict[str, Any], intent_type: str = "unknown") -> ProductEvaluation:
        """Enhanced system output evaluation with comprehensive Slovenian support"""
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
                        item_name="Napaka pri ocenjevanju izdelka",
                        overall_score=50.0,
                        relevance_scores={},
                        issues=["Napaka pri ocenjevanju"],
                        strengths=[],
                        explanation="Napaka med ocenjevanjem"
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