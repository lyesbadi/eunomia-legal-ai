"""
EUNOMIA Legal AI Platform - LLM Service
Ollama EuroLLM-9B service for custom recommendations and clause generation
"""
from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# LLM SERVICE (EUROLLM-9B)
# ============================================================================
class LLMService:
    """
    LLM Service for Ollama EuroLLM-9B interactions.
    
    EuroLLM-9B advantages:
    - 35 languages (all 24 official EU languages + 11 strategic languages)
    - European sovereignty (EU-funded, EU-developed)
    - Superior multilingual performance vs Mistral-7B
    - Excellent French/German/Spanish/Italian capabilities
    - Apache 2.0 license (fully open-source)
    
    Features:
    - Custom legal recommendations
    - Clause generation
    - Legal explanation in plain language
    - Risk analysis narrative
    - Multilingual support (FR, DE, ES, IT, etc.)
    
    Memory: ~5-6 GB (Q4 quantization)
    """
    
    def __init__(self):
        """Initialize LLM service with EuroLLM-9B."""
        self.base_url = settings.OLLAMA_URL
        self.model = settings.OLLAMA_MODEL
        self.temperature = settings.OLLAMA_TEMPERATURE
        self.max_tokens = settings.OLLAMA_MAX_TOKENS
        self.timeout = settings.OLLAMA_TIMEOUT
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        )
        
        logger.info(f"🇪🇺 LLM Service initialized: {self.model} @ {self.base_url}")
        logger.info("🌍 EuroLLM-9B supports 35 languages including all EU official languages")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    # ========================================================================
    # CORE LLM METHODS
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text completion using Ollama EuroLLM-9B.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions (optional)
            temperature: Sampling temperature (optional, overrides default)
            max_tokens: Max tokens to generate (optional, overrides default)
            
        Returns:
            Generated text
        """
        start_time = datetime.now()
        
        try:
            # Prepare request
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                    "top_p": 0.9,  # EuroLLM recommandation
                    "stop": ["<|im_end|>", "<|im_start|>"]  # ChatML stop tokens
                }
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
            
            # Call Ollama API
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ EuroLLM generation completed in {elapsed:.2f}s")
            
            return generated_text
        
        except httpx.HTTPError as e:
            logger.error(f"❌ Ollama HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ LLM generation error: {e}")
            raise
    
    # ========================================================================
    # LEGAL RECOMMENDATIONS
    # ========================================================================
    async def generate_recommendations(
        self,
        document_text: str,
        document_type: str,
        classification: Dict[str, Any],
        unfair_clauses: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate custom legal recommendations using EuroLLM-9B.
        
        Leverages EuroLLM's multilingual capabilities for European legal contexts.
        
        Args:
            document_text: Full document text (truncated if needed)
            document_type: Classified document type
            classification: Classification results
            unfair_clauses: Detected unfair clauses
            risk_assessment: Risk analysis results
            
        Returns:
            Recommendation result with text and action items
        """
        # Build context-aware prompt
        system_prompt = """Tu es EuroLLM, un assistant juridique expert spécialisé dans l'analyse de documents légaux européens.
Tu maîtrises toutes les langues officielles de l'Union Européenne et comprends les spécificités juridiques de chaque pays membre.

Ton rôle est de fournir des recommandations claires, précises et actionnables pour des non-juristes, tout en respectant les nuances juridiques de chaque juridiction européenne.

Règles:
- Utilise un langage clair et accessible
- Fournis des recommandations concrètes et actionnables
- Cite les clauses spécifiques problématiques
- Priorise les risques par gravité
- Sois factuel et objectif
- Mentionne les implications RGPD si pertinent
- Adapte-toi à la langue du document source
"""
        
        # Prepare document summary for context
        doc_summary = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
        
        # Build structured prompt
        unfair_summary = ""
        if unfair_clauses:
            unfair_summary = f"\n\n**Clauses problématiques détectées ({len(unfair_clauses)}):**\n"
            for clause in unfair_clauses[:5]:  # Top 5
                unfair_summary += f"- [{clause['severity'].upper()}] {clause['category']}: {clause['text'][:100]}...\n"
        
        risk_summary = ""
        if risk_assessment:
            risk_summary = f"\n\n**Niveau de risque global:** {risk_assessment.get('risk_level', 'N/A').upper()}"
            risk_summary += f"\n**Score de risque:** {risk_assessment.get('risk_score', 0):.2f}/1.0"
        
        prompt = f"""Analyse le document juridique suivant et fournis des recommandations.

**Type de document:** {document_type}
**Classification:** {classification.get('primary_class', 'unknown')} (confiance: {classification.get('confidence', 0):.0%})
{unfair_summary}
{risk_summary}

**Extrait du document:**
{doc_summary}

**Instructions:**
1. Résume les points clés du document en 2-3 phrases
2. Liste les 3-5 risques principaux identifiés
3. Fournis 3-5 recommandations concrètes et actionnables
4. Indique si une revue par un avocat est nécessaire
5. Mentionne les implications RGPD si pertinent

Format ta réponse en sections claires avec des puces.
"""
        
        # Generate recommendations
        recommendation_text = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        # Extract action items (simple parsing)
        action_items = self._extract_action_items(recommendation_text)
        
        # Determine priority
        priority = "high" if risk_assessment.get('risk_level') in ['high', 'critical'] else "medium"
        
        return {
            "recommendation_text": recommendation_text,
            "action_items": action_items,
            "estimated_priority": priority,
            "confidence": 0.85,  # LLM confidence estimation
            "model_used": self.model
        }
    
    def _extract_action_items(self, text: str) -> List[str]:
        """
        Extract action items from recommendation text.
        
        Simple heuristic: Lines starting with - or numbers.
        
        Args:
            text: Recommendation text
            
        Returns:
            List of action items
        """
        action_items = []
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            # Look for bullet points or numbered lists
            if line_stripped and (
                line_stripped.startswith('-') or
                line_stripped.startswith('•') or
                (len(line_stripped) > 2 and line_stripped[0].isdigit() and line_stripped[1] in '.)')
            ):
                # Remove bullet/number prefix
                item = line_stripped.lstrip('-•0123456789.)').strip()
                if item and len(item) > 10:  # Meaningful action
                    action_items.append(item)
        
        return action_items[:10]  # Max 10 items
    
    # ========================================================================
    # CLAUSE GENERATION
    # ========================================================================
    async def generate_custom_clause(
        self,
        clause_type: str,
        context: str,
        requirements: List[str],
        target_language: str = "fr"
    ) -> str:
        """
        Generate custom legal clause using EuroLLM-9B.
        
        Leverages EuroLLM's multilingual capabilities for European legal clauses.
        
        Args:
            clause_type: Type of clause (e.g., "confidentiality", "termination")
            context: Business context
            requirements: Specific requirements
            target_language: Target language code (fr, de, es, it, etc.)
            
        Returns:
            Generated clause text
        """
        system_prompt = f"""Tu es un juriste expert en rédaction de clauses contractuelles européennes.
Tu rédiges des clauses claires, précises et juridiquement solides en {target_language.upper()}.

Règles:
- Utilise un langage juridique professionnel mais compréhensible
- Inclus les définitions nécessaires
- Précise les délais et modalités
- Anticipe les cas particuliers
- Assure l'équilibre entre les parties
- Respecte les standards juridiques européens
- Mentionne la conformité RGPD si pertinent
"""
        
        requirements_text = "\n".join([f"- {req}" for req in requirements])
        
        prompt = f"""Rédige une clause contractuelle de type "{clause_type}" en {target_language.upper()}.

**Contexte:**
{context}

**Exigences spécifiques:**
{requirements_text}

Rédige une clause complète, structurée et prête à être intégrée dans un contrat.
Assure-toi que la clause respecte les standards juridiques européens et soit conforme au RGPD si applicable.
"""
        
        clause_text = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5  # Lower temperature for legal precision
        )
        
        return clause_text
    
    # ========================================================================
    # LEGAL EXPLANATION (MULTILINGUAL)
    # ========================================================================
    async def explain_legal_concept(
        self,
        concept: str,
        context: str,
        target_audience: str = "non-expert",
        target_language: str = "fr"
    ) -> str:
        """
        Explain legal concept in plain language using EuroLLM-9B.
        
        Supports all 35 EuroLLM languages for European legal contexts.
        
        Args:
            concept: Legal concept to explain
            context: Context from document
            target_audience: Target audience (non-expert, business, legal)
            target_language: Target language code (fr, de, es, it, etc.)
            
        Returns:
            Explanation text in target language
        """
        audience_instructions = {
            "non-expert": f"Explique comme si tu parlais à quelqu'un sans connaissance juridique en {target_language.upper()}. Utilise des exemples concrets.",
            "business": f"Explique en termes business en {target_language.upper()} avec focus sur les implications pratiques et financières.",
            "legal": f"Fournis une explication technique précise en {target_language.upper()} avec références juridiques pertinentes."
        }
        
        system_prompt = f"""Tu es un expert juridique européen qui vulgarise le droit dans toutes les langues de l'UE.
{audience_instructions.get(target_audience, audience_instructions['non-expert'])}

Ton explication doit être:
- Claire et accessible
- Précise et factuelle
- Illustrée par des exemples européens pertinents
- Structurée logiquement
- Adaptée aux spécificités juridiques du pays concerné
"""
        
        prompt = f"""Explique le concept juridique suivant en {target_language.upper()}:

**Concept:** {concept}

**Contexte dans le document:**
{context[:500]}

Fournis une explication complète en 3-5 paragraphes.
Si le concept a des implications RGPD, mentionne-les.
"""
        
        explanation = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6
        )
        
        return explanation
    
    # ========================================================================
    # RISK NARRATIVE
    # ========================================================================
    async def generate_risk_narrative(
        self,
        risk_factors: List[Dict[str, Any]],
        risk_level: str,
        document_type: str,
        target_language: str = "fr"
    ) -> str:
        """
        Generate narrative explanation of risks using EuroLLM-9B.
        
        Args:
            risk_factors: List of detected risk factors
            risk_level: Overall risk level
            document_type: Type of document
            target_language: Target language code
            
        Returns:
            Risk narrative text
        """
        system_prompt = f"""Tu es un analyste de risques juridiques européen.
Tu expliques les risques de manière claire et structurée en {target_language.upper()}.

Ton analyse doit:
- Prioriser les risques par gravité
- Expliquer les implications concrètes dans le contexte européen
- Suggérer des mesures d'atténuation conformes au RGPD
- Rester factuelle et objective
"""
        
        risks_text = "\n".join([
            f"- [{rf.get('severity', 'N/A').upper()}] {rf.get('factor', 'N/A')}: {rf.get('description', 'N/A')}"
            for rf in risk_factors[:10]
        ])
        
        prompt = f"""Analyse les risques suivants pour un document de type "{document_type}" en {target_language.upper()}:

**Niveau de risque global:** {risk_level.upper()}

**Facteurs de risque identifiés:**
{risks_text}

Rédige une analyse narrative structurée en:
1. Synthèse du niveau de risque
2. Risques majeurs (top 3)
3. Implications concrètes dans le contexte juridique européen
4. Recommandations de mitigation conformes au RGPD

Format: 4-6 paragraphes, ton professionnel mais accessible.
"""
        
        narrative = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        return narrative
    
    # ========================================================================
    # HEALTH CHECK
    # ========================================================================
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Ollama service health.
        
        Returns:
            Health status dictionary
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_available = any(m.get("name") == self.model for m in models)
            
            return {
                "status": "healthy",
                "ollama_url": self.base_url,
                "model": self.model,
                "model_available": model_available,
                "available_models": [m.get("name") for m in models],
                "multilingual_support": "35 languages (all 24 EU official languages)",
                "european_sovereignty": True
            }
        
        except Exception as e:
            logger.error(f"❌ Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
llm_service = LLMService()