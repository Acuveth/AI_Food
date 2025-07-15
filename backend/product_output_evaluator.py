#!/usr/bin/env python3
"""
Simple LLM-Based Output Validator
Uses only LLM to determine if output matches user intent and improves it if needed
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMOutputValidator:
    """Simple LLM-based output validator"""
    
    def __init__(self):
        self.client = client
        
        # Validation prompt template
        self.validation_prompt = """
        Analiziraj ali se rezultat ujema z uporabnikovo zahtevo:

        UPORABNIKOVA ZAHTEVA: "{user_input}"
        
        REZULTAT SISTEMA: {system_output}
        
        POMEMBNO: POGLEJ ČE JE REZULTAT SISTEMA SPLOH TISTA STVAR, KI JO UPORABNIK IŠČE ALI JE TO SAMO EN DRUG PRODUKT S PODOBNIM IMENOM, KI SPLOH NE USTREZA ZAHTEVI UPORABNIKA

        Evalviraj:
        1. Ali rezultat odgovarja na uporabnikovo zahtevo?
        2. Ali so prikazani izdelki/jedi relevantni?
        3. Ali so podatki koristni za uporabnika?
        4. Ali manjkajo pomembne informacije?
        
        Odgovori z JSON:
        {{
            "is_relevant": true/false,
            "relevance_score": 0-100,
            "issues": ["seznam problemov"],
            "improvements_needed": ["seznam izboljšav"],
            "action": "keep_as_is" | "filter_results" | "add_explanation" | "completely_different_search",
            "filtered_indices": [indeksi za obdržanje - samo če action="filter_results"],
            "additional_explanation": "dodaten opis za uporabnika - samo če action="add_explanation"",
            "alternative_suggestion": "alternativni predlog - samo če action="completely_different_search"",
            "reasoning": "zakaj je sprejeta ta odločitev"
        }}
        """
    
    async def validate_and_improve_output(
        self, 
        user_input: str, 
        potential_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main validation function - uses LLM to validate and improve output
        """
        logger.info(f"🔍 Validating output for: '{user_input[:50]}...'")
        
        try:
            # Use LLM to evaluate the output
            evaluation = await self._evaluate_output_with_llm(user_input, potential_output)
            
            if not evaluation["is_relevant"] or evaluation["relevance_score"] < 60:
                logger.warning(f"⚠️ Low relevance output detected: {evaluation['relevance_score']}/100")
                return await self._improve_output(potential_output, evaluation, user_input)
            
            # Output is good, maybe add explanation if needed
            if evaluation["action"] == "add_explanation":
                return self._add_explanation(potential_output, evaluation)
            
            logger.info(f"✅ Output validated successfully: {evaluation['relevance_score']}/100")
            return potential_output
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            # Return original output if validation fails
            return potential_output
    
    async def _evaluate_output_with_llm(
        self, 
        user_input: str, 
        system_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to evaluate if output matches user intent"""
        
        # Prepare system output for LLM analysis
        output_summary = self._summarize_output_for_llm(system_output)
        
        prompt = self.validation_prompt.format(
            user_input=user_input,
            system_output=output_summary
        )
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
            
            evaluation = json.loads(json_text)
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM evaluation: {e}")
            # Return default "good" evaluation if parsing fails
            return {
                "is_relevant": True,
                "relevance_score": 75,
                "issues": [],
                "improvements_needed": [],
                "action": "keep_as_is",
                "reasoning": "Failed to parse evaluation, assuming good"
            }
    
    def _summarize_output_for_llm(self, output: Dict[str, Any]) -> str:
        """Create a summary of the output for LLM analysis"""
        summary_parts = []
        
        # Check different output types and summarize
        if "promotions" in output:
            promotions = output["promotions"]
            summary_parts.append(f"AKCIJE: {len(promotions)} izdelkov")
            if promotions:
                sample_items = [p.get("product_name", "Unknown") for p in promotions[:3]]
                summary_parts.append(f"Primeri: {', '.join(sample_items)}")
                stores = list(set(p.get("store_name", "") for p in promotions))
                summary_parts.append(f"Trgovine: {', '.join(stores)}")
        
        elif "meals" in output:
            meals = output["meals"]
            summary_parts.append(f"JEDI: {len(meals)} receptov")
            if meals:
                sample_meals = [m.get("title", "Unknown") for m in meals[:3]]
                summary_parts.append(f"Primeri: {', '.join(sample_meals)}")
                cuisines = list(set(m.get("cuisine_type", "") for m in meals if m.get("cuisine_type")))
                if cuisines:
                    summary_parts.append(f"Kuhinje: {', '.join(cuisines[:3])}")
        
        elif "results_by_store" in output:
            stores_data = output["results_by_store"]
            total_products = sum(len(store.get("products", [])) for store in stores_data.values())
            summary_parts.append(f"PRIMERJAVA CEN: {total_products} izdelkov")
            store_names = list(stores_data.keys())
            summary_parts.append(f"Trgovine: {', '.join(store_names)}")
        
        elif "suggested_meals" in output:
            suggested = output["suggested_meals"]
            summary_parts.append(f"PREDLAGANE JEDI: {len(suggested)} jedi")
            ingredients = output.get("available_ingredients", [])
            summary_parts.append(f"Z sestavinami: {', '.join(ingredients[:3])}")
        
        elif "grocery_analysis" in output:
            summary_parts.append("ANALIZA STROŠKOV NAKUPOVANJA")
            meal = output.get("meal", {})
            meal_title = meal.get("title", "Unknown meal")
            summary_parts.append(f"Za jed: {meal_title}")
        
        else:
            # General response
            response_text = output.get("response", "")
            summary_parts.append(f"SPLOŠEN ODGOVOR: {response_text[:100]}...")
        
        return " | ".join(summary_parts)
    
    async def _improve_output(
        self, 
        original_output: Dict[str, Any], 
        evaluation: Dict[str, Any], 
        user_input: str
    ) -> Dict[str, Any]:
        """Improve output based on LLM evaluation"""
        
        action = evaluation.get("action", "keep_as_is")
        
        if action == "filter_results":
            return self._filter_results(original_output, evaluation)
        
        elif action == "add_explanation":
            return self._add_explanation(original_output, evaluation)
        
        elif action == "completely_different_search":
            return self._suggest_alternative(original_output, evaluation, user_input)
        
        else:
            # Default: keep as is but add issues note
            return self._add_issues_note(original_output, evaluation)
    
    def _filter_results(self, output: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Filter results based on LLM-suggested indices"""
        indices_to_keep = evaluation.get("filtered_indices", [])
        
        if not indices_to_keep:
            # If no indices specified, keep first half of results
            if "promotions" in output:
                original_count = len(output["promotions"])
                indices_to_keep = list(range(min(5, original_count // 2)))
            elif "meals" in output:
                original_count = len(output["meals"])
                indices_to_keep = list(range(min(5, original_count // 2)))
        
        filtered_output = output.copy()
        
        # Filter promotions
        if "promotions" in output:
            original_promotions = output["promotions"]
            filtered_promotions = [original_promotions[i] for i in indices_to_keep if i < len(original_promotions)]
            filtered_output["promotions"] = filtered_promotions
            filtered_output["total_found"] = len(filtered_promotions)
            
            # Update summary
            filtered_output["summary"] = f"Filtrirani rezultati: {len(filtered_promotions)} najbolj relevantnih izdelkov"
        
        # Filter meals
        elif "meals" in output:
            original_meals = output["meals"]
            filtered_meals = [original_meals[i] for i in indices_to_keep if i < len(original_meals)]
            filtered_output["meals"] = filtered_meals
            filtered_output["final_count"] = len(filtered_meals)
            
            # Update summary
            filtered_output["summary"] = f"Filtrirani rezultati: {len(filtered_meals)} najbolj ustreznih jedi"
        
        logger.info(f"🔧 Filtered results: kept {len(indices_to_keep)} most relevant items")
        return filtered_output
    
    def _add_explanation(self, output: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional explanation to help user"""
        explanation = evaluation.get("additional_explanation", "")
        
        if not explanation:
            return output
        
        enhanced_output = output.copy()
        
        # Add explanation to summary
        current_summary = enhanced_output.get("summary", "")
        enhanced_summary = f"{current_summary}\n\n💡 {explanation}"
        enhanced_output["summary"] = enhanced_summary
        
        logger.info(f"📝 Added explanation to output")
        return enhanced_output
    
    def _suggest_alternative(self, output: Dict[str, Any], evaluation: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Suggest completely different search when current results are irrelevant"""
        alternative = evaluation.get("alternative_suggestion", "")
        
        alternative_output = {
            "success": True,
            "message": f"Vaša zahteva '{user_input}' ni dala dobrih rezultatov.",
            "data": {
                "response": f"Ni najdenih ustreznih rezultatov za vašo zahtevo. {alternative}",
                "suggestions": [
                    "Poskusite z drugačnimi iskalnimi pojmi",
                    "Uporabite slovenska imena izdelkov",
                    "Omenite lahko trgovino (DM, Lidl, Mercator, SPAR, Tuš)",
                    alternative if alternative else "Poskusite bolj specifične izraze"
                ]
            },
            "alternative_suggestions": True
        }
        
        logger.info(f"🔄 Suggested alternative search approach")
        return alternative_output
    
    def _add_issues_note(self, output: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Add note about potential issues without major changes"""
        issues = evaluation.get("issues", [])
        
        if not issues:
            return output
        
        enhanced_output = output.copy()
        current_summary = enhanced_output.get("summary", "")
        
        # Add subtle note about limitations
        issue_note = "Opomba: Rezultati morda niso popolnoma ustrezni vaši zahtevi."
        enhanced_summary = f"{current_summary}\n\n⚠️ {issue_note}"
        enhanced_output["summary"] = enhanced_summary
        
        return enhanced_output

# Global validator instance
llm_validator = LLMOutputValidator()

async def validate_output(user_input: str, potential_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main validation function - validates and improves any system output
    """
    return await llm_validator.validate_and_improve_output(user_input, potential_output)