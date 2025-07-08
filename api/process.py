# Process.py V23 HYBRIDE - Système actuel + Cognee en complément
# Garde votre mémoire ConversationBufferMemory ET ajoute Cognee

import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
import json
import re

# NOUVEAU: Import Cognee
try:
    import cognee
    import asyncio
    COGNEE_AVAILABLE = True
    print("✅ Cognee disponible")
except ImportError:
    COGNEE_AVAILABLE = False
    print("⚠️ Cognee non disponible - mode fallback")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JAK Company AI Agent API HYBRIDE", version="23.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if COGNEE_AVAILABLE:
    os.environ["LLM_API_KEY"] = os.getenv("OPENAI_API_KEY")

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is required")

# Store pour la mémoire EXISTANTE (on garde !)
memory_store: Dict[str, ConversationBufferMemory] = {}

# NOUVEAU: Gestionnaire Cognee HYBRIDE
class HybridCogneeManager:
    """Gestionnaire hybride qui combine votre système actuel avec Cognee"""
    
    def __init__(self):
        self.cognee_initialized = False
        self.fallback_mode = not COGNEE_AVAILABLE
        
    async def initialize_cognee_if_available(self):
        """Initialise Cognee seulement si disponible"""
        if not COGNEE_AVAILABLE:
            logger.info("📋 Cognee non disponible - utilisation système existant uniquement")
            return
            
        try:
            # Ajouter la base de connaissances JAK Company
            await self._populate_jak_knowledge()
            self.cognee_initialized = True
            logger.info("✅ Cognee initialisé en mode hybride")
        except Exception as e:
            logger.error(f"❌ Erreur init Cognee: {str(e)}")
            self.fallback_mode = True
    
    async def _populate_jak_knowledge(self):
        """Peuple Cognee avec la base de connaissances JAK"""
        
        # Base de connaissances extraite de votre BDD actuelle
        jak_knowledge_chunks = [
            # Paiements et délais
            """JAK Company - Règles de paiement selon financement:
            CPF: Délai minimum 45 jours après réception feuilles émargement signées. 
            Depuis février 2025, moins de 50 dossiers sur 2500 sont bloqués par la réforme Caisse des Dépôts.
            OPCO: Délai moyen 2 mois, maximum 6 mois selon organisme.
            Financement direct: 7 jours après fin formation et réception dossier complet.""",
            
            # Programme ambassadeur 
            """Programme Ambassadeur JAK Company:
            Étape 1: S'abonner Instagram (https://hi.switchy.io/InstagramWeiWei) et Snapchat (https://hi.switchy.io/SnapChatWeiWei)
            Étape 2: Créer code affiliation sur https://swiy.co/jakpro
            Étape 3: Envoyer contacts via https://mrqz.to/AffiliationPromotion (nom, prénom, téléphone, SIRET si entreprise)
            Étape 4: Commission jusqu'à 60% si dossier validé. Paiement possible compte perso jusqu'à 3000€/an et 3 virements.""",
            
            # Formations disponibles
            """JAK Company Formations:
            Plus de 100 formations: Bureautique (Word, Excel, PowerPoint), Informatique, Développement Web/3D, 
            Langues étrangères, Vente & Marketing digital, Développement personnel, Écologie & Numérique responsable, 
            Bilan de compétences. IMPORTANT: Plus de formations CPF depuis février 2025.""",
            
            # Support et horaires
            """Support JAK Company:
            Horaires équipe: Lundi-Vendredi 9h-17h (hors pause déjeuner)
            Escalade ADMIN: Paiements, problèmes techniques, dossiers CPF
            Escalade FORMATION: Formations professionnels/particuliers  
            Escalade ENTREPRISE: Demandes B2B
            Réseaux sociaux pour actualités: Instagram et Snapchat"""
        ]
        
        # Ajouter chaque chunk à Cognee
        for i, chunk in enumerate(jak_knowledge_chunks):
            await cognee.add(chunk, dataset_name=f"jak_knowledge_{i}")
        
        # Générer le knowledge graph
        await cognee.cognify()
        logger.info("📚 Base JAK ajoutée à Cognee")
    
    async def try_cognee_search(self, user_message: str, wa_id: str) -> Optional[Dict[str, Any]]:
        """Essaie une recherche Cognee si disponible"""
        
        if not COGNEE_AVAILABLE or not self.cognee_initialized or self.fallback_mode:
            return None
            
        try:
            # Recherche dans Cognee
            results = await cognee.search(user_message, user=wa_id)
            
            if not results or len(results) == 0:
                return None
                
            # Analyser la pertinence
            confidence = min(len(results) / 5.0, 1.0)  # Plus de résultats = plus de confiance
            
            # Seuil de confiance conservateur
            if confidence < 0.3:
                return None
                
            # Formater la réponse
            main_result = str(results[0]) if results else ""
            if len(main_result) > 500:
                main_result = main_result[:500] + "..."
                
            return {
                "response": main_result,
                "confidence": confidence,
                "results_count": len(results),
                "source": "cognee"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur Cognee search: {str(e)}")
            return None

# Instance globale du gestionnaire hybride
cognee_manager = HybridCogneeManager()

# GARDE VOTRE CLASSE EXISTANTE (inchangée)
class MemoryManager:
    """Gestionnaire de mémoire optimisé - CONSERVÉ TEL QUEL"""
    
    @staticmethod
    def trim_memory(memory: ConversationBufferMemory, max_messages: int = 15):
        messages = memory.chat_memory.messages
        if len(messages) > max_messages:
            memory.chat_memory.messages = messages[-max_messages:]
            logger.info(f"Memory trimmed to {max_messages} messages")
    
    @staticmethod
    def get_memory_summary(memory: ConversationBufferMemory) -> Dict[str, Any]:
        messages = memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if hasattr(m, 'type') and m.type == 'human']),
            "ai_messages": len([m for m in messages if hasattr(m, 'type') and m.type == 'ai']),
            "memory_size_chars": sum(len(str(m.content)) for m in messages)
        }

# GARDE VOS CLASSES EXISTANTES (PaymentContextProcessor, MessageProcessor, etc.)
# Je vais juste ajouter la logique Cognee dans l'endpoint principal

# Vos classes existantes conservées...
class PaymentContextProcessor:
    """CONSERVÉ EXACTEMENT TEL QUEL - Votre logique existante"""
    
    @staticmethod
    def extract_financing_type(message: str) -> Optional[str]:
        # Votre code existant inchangé
        message_lower = message.lower()
        
        financing_patterns = {
            'CPF': ['cpf', 'compte personnel', 'compte personnel formation'],
            'OPCO': ['opco', 'operateur', 'opérateur', 'organisme paritaire'],
            'direct': ['en direct', 'financé en direct', 'financement direct', 'entreprise', 'particulier']
        }
        
        for financing_type, patterns in financing_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return financing_type
        return None
    
    @staticmethod
    def extract_time_delay(message: str) -> Optional[int]:
        # Votre code existant inchangé
        message_lower = message.lower()
        
        delay_patterns = [
            r'(?:il y a|depuis|ça fait|ca fait)\s*(\d+)\s*mois',
            r'(?:il y a|depuis|ça fait|ca fait)\s*(\d+)\s*semaines?',
            r'(\d+)\s*mois',
        ]
        
        for pattern in delay_patterns:
            match = re.search(pattern, message_lower)
            if match:
                number = int(match.group(1))
                if 'semaine' in pattern:
                    return max(1, round(number / 4.33))
                return number
        return None

class MessageProcessor:
    """CONSERVÉ + AJOUT logique hybride Cognee"""
    
    @staticmethod
    async def detect_priority_rules_hybrid(user_message: str, matched_bloc_response: str,
                                         conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Version hybride qui essaie Cognee PUIS votre système existant"""
        
        # NOUVEAU: Essayer Cognee d'abord pour les cas complexes
        cognee_result = await cognee_manager.try_cognee_search(user_message, 
                                                             conversation_context.get("wa_id", "unknown"))
        
        if cognee_result and cognee_result["confidence"] > 0.5:
            logger.info(f"✅ Réponse Cognee trouvée (conf: {cognee_result['confidence']:.2f})")
            return {
                "use_matched_bloc": False,
                "priority_detected": "COGNEE_RESPONSE",
                "response": cognee_result["response"],
                "confidence": cognee_result["confidence"],
                "source": "cognee",
                "context": conversation_context
            }
        
        # FALLBACK: Utiliser VOTRE SYSTÈME EXISTANT (code original)
        logger.info("📋 Fallback vers système existant")
        
        # Votre logique de détection prioritaire existante (inchangée)
        message_lower = user_message.lower()
        
        # Votre logique agressivité
        aggressive_terms = ["merde", "nul", "batard", "enervez"]
        if any(term in message_lower for term in aggressive_terms):
            return {
                "use_matched_bloc": False,
                "priority_detected": "AGRESSIVITE",
                "response": "Être impoli ne fera pas avancer la situation plus vite. Bien au contraire. Souhaites-tu que je te propose un poème ou une chanson d'amour pour apaiser ton cœur ? 💌",
                "context": conversation_context,
                "source": "existing_system"
            }
        
        # Votre logique paiement formation
        payment_keywords = ["pas été payé", "rien reçu", "virement", "attends", "paiement", "argent"]
        if any(keyword in message_lower for keyword in payment_keywords):
            financing_type = PaymentContextProcessor.extract_financing_type(user_message)
            delay_months = PaymentContextProcessor.extract_time_delay(user_message)
            
            if financing_type and delay_months:
                if financing_type == "CPF" and delay_months >= 2:
                    return {
                        "use_matched_bloc": False,
                        "priority_detected": "CPF_DELAI_DEPASSE_FILTRAGE",
                        "response": """Juste avant que je transmette ta demande 🙏

Est-ce que tu as déjà été informé par l'équipe que ton dossier CPF faisait partie des quelques cas bloqués par la Caisse des Dépôts ?

👉 Si oui, je te donne directement toutes les infos liées à ce blocage.
Sinon, je fais remonter ta demande à notre équipe pour vérification ✅""",
                        "context": conversation_context,
                        "awaiting_cpf_info": True,
                        "source": "existing_system"
                    }
        
        # Si un bloc est fourni par n8n, l'utiliser
        if matched_bloc_response and matched_bloc_response.strip():
            return {
                "use_matched_bloc": True,
                "priority_detected": "N8N_BLOC_DETECTED",
                "response": matched_bloc_response,
                "context": conversation_context,
                "source": "existing_system"
            }
        
        # Fallback général
        return {
            "use_matched_bloc": False,
            "priority_detected": "FALLBACK_GENERAL",
            "context": conversation_context,
            "response": None,
            "use_ai": True,
            "source": "existing_system"
        }

# ENDPOINT PRINCIPAL MODIFIÉ (Hybride)
@app.post("/")
async def process_message_hybrid(request: Request):
    """Point d'entrée HYBRIDE - Cognee + votre système existant"""
    
    try:
        # Parse request (inchangé)
        body = await request.json()
        
        user_message = body.get("message_original", body.get("message", ""))
        matched_bloc_response = body.get("matched_bloc_response", "")
        wa_id = body.get("wa_id", "default_wa_id")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"[{wa_id}] HYBRID Processing: '{user_message[:50]}...'")
        
        # Gestion de la mémoire EXISTANTE (inchangée)
        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        
        memory = memory_store[wa_id]
        MemoryManager.trim_memory(memory, max_messages=15)
        
        # Analyser le contexte avec votre système existant
        conversation_context = {
            "message_count": len(memory.chat_memory.messages),
            "wa_id": wa_id,
            "is_follow_up": len(memory.chat_memory.messages) > 0
        }
        
        # Ajouter le message utilisateur à la mémoire
        memory.chat_memory.add_user_message(user_message)
        
        # NOUVEAU: Traitement hybride
        priority_result = await MessageProcessor.detect_priority_rules_hybrid(
            user_message, matched_bloc_response, conversation_context
        )
        
        # Construire la réponse
        final_response = priority_result.get("response")
        response_source = priority_result.get("source", "unknown")
        
        if not final_response:
            final_response = matched_bloc_response or """Salut 👋

Je vais faire suivre ta demande à notre équipe ! 😊

🕐 Notre équipe est disponible du lundi au vendredi, de 9h à 17h.
On te tiendra informé dès que possible ✅"""
            response_source = "fallback"
        
        # Ajouter à la mémoire
        memory.chat_memory.add_ai_message(final_response)
        MemoryManager.trim_memory(memory, max_messages=15)
        
        # Ajouter la conversation à Cognee pour apprentissage (si disponible)
        if COGNEE_AVAILABLE and cognee_manager.cognee_initialized:
            try:
                conversation_data = f"Utilisateur {wa_id}: {user_message} | Réponse: {final_response}"
                await cognee.add(conversation_data, dataset_name=f"conversations_{wa_id}")
            except Exception as e:
                logger.error(f"Erreur ajout Cognee: {str(e)}")
        
        # Réponse avec métadonnées
        return {
            "matched_bloc_response": final_response,
            "confidence": priority_result.get("confidence", 0.7),
            "processing_type": priority_result.get("priority_detected", "hybrid"),
            "escalade_required": priority_result.get("escalade_required", False),
            "escalade_type": priority_result.get("escalade_type", "general"),
            "status": "hybrid_success",
            "response_source": response_source,
            "cognee_available": COGNEE_AVAILABLE,
            "session_id": wa_id,
            "memory_summary": MemoryManager.get_memory_summary(memory)
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid processing: {str(e)}")
        return {
            "matched_bloc_response": """Salut 👋

Je rencontre un petit problème technique. Notre équipe va regarder ça ! 😊

🕐 Horaires : Lundi-Vendredi, 9h-17h""",
            "confidence": 0.1,
            "processing_type": "error_fallback",
            "escalade_required": True,
            "status": "error",
            "response_source": "error_fallback"
        }

# Endpoints de gestion (nouveaux)
@app.get("/health")
async def health_check():
    """Status du système hybride"""
    return {
        "status": "healthy",
        "version": "23.0 HYBRID",
        "cognee_available": COGNEE_AVAILABLE,
        "cognee_initialized": cognee_manager.cognee_initialized if COGNEE_AVAILABLE else False,
        "fallback_mode": cognee_manager.fallback_mode if COGNEE_AVAILABLE else True,
        "memory_system": "ConversationBufferMemory + Cognee",
        "active_sessions": len(memory_store)
    }

@app.post("/cognee/reset")
async def reset_cognee():
    """Reset Cognee uniquement"""
    if not COGNEE_AVAILABLE:
        raise HTTPException(status_code=400, detail="Cognee not available")
    
    try:
        await cognee.reset()
        await cognee_manager.initialize_cognee_if_available()
        return {"status": "Cognee reset successfully", "memory_store": "preserved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialisation au démarrage
@app.on_event("startup")
async def startup_event():
    """Initialise Cognee si disponible"""
    if COGNEE_AVAILABLE:
        await cognee_manager.initialize_cognee_if_available()
    else:
        logger.info("📋 Démarrage en mode système existant uniquement")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)