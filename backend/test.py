#!/usr/bin/env python3
"""
Test script to verify the bug fixes work properly
"""

import asyncio
import logging
import json
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_handler():
    """Test the database handler fixes"""
    try:
        from database_handler import get_db_handler
        
        print("🧪 Testing Database Handler...")
        
        # Test database connection
        db_handler = await get_db_handler()
        print("✅ Database connection successful")
        
        # Test decimal conversion
        test_row = {
            "product_name": "Test Product",
            "current_price": Decimal("2.50"),
            "regular_price": Decimal("3.00"),
            "has_discount": True,
            "discount_percentage": 17
        }
        
        converted = db_handler._convert_decimals(test_row)
        print(f"✅ Decimal conversion test: {converted}")
        print(f"   Price type: {type(converted['current_price'])}")
        
        # Test search term generation
        search_terms = await db_handler._generate_item_search_terms("milk")
        print(f"✅ Search term generation: {search_terms}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database handler test failed: {e}")
        return False

async def test_meal_search():
    """Test the meal search fixes"""
    try:
        from meal_search import MealSearcher
        
        print("\n🧪 Testing Meal Search...")
        
        meal_searcher = MealSearcher()
        
        # Test decimal handling in cost analysis
        test_ingredient_results = {
            "milk": [
                {"store_name": "dm", "current_price": Decimal("1.50"), "product_name": "Test Milk"},
                {"store_name": "lidl", "current_price": Decimal("1.30"), "product_name": "Lidl Milk"}
            ],
            "bread": [
                {"store_name": "mercator", "current_price": Decimal("2.00"), "product_name": "Test Bread"}
            ]
        }
        
        store_analysis = meal_searcher._analyze_store_costs(test_ingredient_results)
        print(f"✅ Store analysis: {store_analysis['dm']['total_cost']}")
        print(f"   Cost type: {type(store_analysis['dm']['total_cost'])}")
        
        combined_analysis = meal_searcher._analyze_combined_cheapest(test_ingredient_results)
        print(f"✅ Combined analysis: {combined_analysis['total_cost']}")
        print(f"   Cost type: {type(combined_analysis['total_cost'])}")
        
        # Test meal statistics calculation
        test_meal_data = {"title": "Test Meal", "servings": 4}
        stats = meal_searcher._calculate_meal_statistics(test_meal_data, test_ingredient_results, combined_analysis)
        print(f"✅ Meal statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Meal search test failed: {e}")
        return False

def test_json_parsing():
    """Test JSON parsing scenarios"""
    print("\n🧪 Testing JSON Parsing...")
    
    # Simulate different LLM response formats
    test_responses = [
        '{"search_terms": ["milk", "mleko"], "category_hints": ["dairy"]}',
        '```json\n{"search_terms": ["bread", "kruh"]}\n```',
        'Here are the search terms: {"search_terms": ["cheese", "sir"]}',
        'Invalid response without JSON',
        '{"search_terms": ["pasta", "testenine"], "malformed": }'
    ]
    
    for i, response in enumerate(test_responses):
        try:
            # Simulate the parsing logic
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith('{'):
                json_text = response
            else:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = response[start_idx:end_idx]
                else:
                    print(f"❌ Test {i+1}: No JSON found")
                    continue
            
            json_text = json_text.replace('\n', ' ').replace('\r', ' ').strip()
            result = json.loads(json_text)
            search_terms = result.get("search_terms", ["fallback"])
            print(f"✅ Test {i+1}: {search_terms}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Test {i+1}: JSON error - {e}")
        except Exception as e:
            print(f"❌ Test {i+1}: Other error - {e}")

async def test_input_interpreter():
    """Test the input interpreter"""
    try:
        from input_interpreter import interpret_user_input
        
        print("\n🧪 Testing Input Interpreter...")
        
        test_inputs = [
            "Find milk deals",
            "Compare bread prices",
            "Healthy Italian dinner recipes",
            "What can I cook with chicken and rice?"
        ]
        
        for test_input in test_inputs:
            try:
                result = await interpret_user_input(test_input)
                print(f"✅ '{test_input}' → Intent: {result['intent']}")
            except Exception as e:
                print(f"❌ '{test_input}' → Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Input interpreter test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting bug fix tests...\n")
    
    # Test JSON parsing (doesn't require database)
    test_json_parsing()
    
    # Test input interpreter (requires OpenAI API)
    try:
        await test_input_interpreter()
    except Exception as e:
        print(f"⚠️ Input interpreter test skipped: {e}")
    
    # Test database handler (requires database connection)
    try:
        await test_database_handler()
    except Exception as e:
        print(f"⚠️ Database test skipped: {e}")
    
    # Test meal search
    try:
        await test_meal_search()
    except Exception as e:
        print(f"⚠️ Meal search test skipped: {e}")
    
    print("\n🏁 Tests completed!")
    print("\n💡 To test the full system:")
    print("1. Run: python streamlined_backend.py")
    print("2. Open frontend and try: 'Healthy Italian dinner recipes'")
    print("3. Select a meal to test grocery analysis")

if __name__ == "__main__":
    asyncio.run(main())