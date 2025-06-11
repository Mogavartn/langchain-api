import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
import json
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JAK Company AI Agent API", version="4.0")

# Configuration CORS pour permettre les tests locaux
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vérification de la clé API OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# Store pour la mémoire des conversations
memory_store: Dict[str, ConversationBufferMemory] = {}

class ResponseValidator:
    """Classe pour valider et nettoyer les réponses"""
    
    @staticmethod
    def clean_response(response: str) -> str:
        """Nettoie et formate la réponse"""
        if not response:
            return ""
        
        # Supprimer les caractères de contrôle
        response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
        
        # Nettoyer les espaces multiples
        response = re.sub(r'\s+', ' ', response.strip())
        
        return response
    
    @staticmethod
    def validate_escalade_keywords(message: str) -> Optional[str]:
        """Détecte si le message nécessite une escalade"""
        escalade_keywords = [
            "retard anormal", "paiement bloqué", "problème grave",
            "urgence", "plainte", "avocat", "tribunal"
        ]
        
        message_lower = message.lower()
        for keyword in escalade_keywords:
            if keyword in message_lower:
                return "admin"
        
        return None

class MessageProcessor:
    """Classe principale pour traiter les messages"""
    
    @staticmethod
    def detect_priority_rules(user_message: str, matched_bloc_response: str) -> Dict[str, Any]:
        """Applique les règles de priorité selon la BDD"""
        
        message_lower = user_message.lower()
        
        # RÈGLE 1: Détection problème paiement formation (PRIORITÉ ABSOLUE)
        payment_keywords = [
            "pas été payé", "rien reçu", "virement", "attends", 
            "paiement", "argent", "retard", "promesse"
        ]
        
        if any(keyword in message_lower for keyword in payment_keywords):
            # Si un bloc est déjà matché pour le paiement, le garder
            if matched_bloc_response and ("paiement" in matched_bloc_response.lower() or "délai" in matched_bloc_response.lower()):
                return {
                    "use_matched_bloc": True,
                    "priority_detected": "PAIEMENT_FORMATION",
                    "response": matched_bloc_response
                }
        
        # RÈGLE 2: Agressivité détectée
        aggressive_keywords = ["merde", "con", "nul", "énervez", "batards"]
        if any(keyword in message_lower for keyword in aggressive_keywords):
            return {
                "use_matched_bloc": False,
                "priority_detected": "AGRESSIVITE",
                "response": "Être impoli ne fera pas avancer la situation plus vite. Bien au contraire. Souhaites-tu que je te propose un poème ou une chanson d'amour pour apaiser ton cœur ? 💌"
            }
        
        # RÈGLE 3: Si matched_bloc_response existe, l'utiliser (REPRODUCTION EXACTE)
        if matched_bloc_response and matched_bloc_response.strip():
            return {
                "use_matched_bloc": True,
                "priority_detected": "BLOC_MATCHE",
                "response": matched_bloc_response
            }
        
        # RÈGLE 4: Escalade automatique si nécessaire
        escalade_type = ResponseValidator.validate_escalade_keywords(user_message)
        if escalade_type:
            return {
                "use_matched_bloc": False,
                "priority_detected": "ESCALADE_AUTO",
                "escalade_type": escalade_type,
                "response": "🔁 ESCALADE AGENT ADMIN\n\n📅 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\n🕐 On te tiendra informé dès qu'on a du nouveau ✅"
            }
        
        return {"use_matched_bloc": False, "priority_detected": "NONE"}

@app.post("/")
async def process_message(request: Request):
    """Point d'entrée principal pour traiter les messages"""
    try:
        body = await request.json()
        
        # Extraction des données avec fallbacks
        user_message = body.get("message_original", body.get("message", ""))
        matched_bloc_response = body.get("matched_bloc_response", "")
        wa_id = body.get("wa_id", "default_wa_id")
        
        logger.info(f"[{wa_id}] Processing: message='{user_message[:50]}...', has_bloc={bool(matched_bloc_response)}")
        
        # Validation des entrées
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Nettoyage des données
        user_message = ResponseValidator.clean_response(user_message)
        matched_bloc_response = ResponseValidator.clean_response(matched_bloc_response)
        
        # Application des règles de priorité
        priority_result = MessageProcessor.detect_priority_rules(user_message, matched_bloc_response)
        
        # Gestion de la mémoire conversation
        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        
        memory = memory_store[wa_id]
        memory.chat_memory.add_user_message(user_message)
        
        # Construction de la réponse selon la priorité
        if priority_result.get("use_matched_bloc") and priority_result.get("response"):
            final_response = priority_result["response"]
            response_type = "exact_match_enforced"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "AGRESSIVITE":
            final_response = priority_result["response"]
            response_type = "agressivite_detected"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "ESCALADE_AUTO":
            final_response = priority_result["response"]
            response_type = "auto_escalade"
            escalade_required = True
            
        else:
            # Fallback généralisé mais informatif
            final_response = """Salut 👋
            
Je vais faire suivre ta demande à notre équipe pour qu'elle puisse t'aider au mieux 😊

🕐 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner).
On te tiendra informé dès que possible ✅

En attendant, peux-tu me préciser un peu plus ce que tu recherches ?"""
            response_type = "fallback_with_context"
            escalade_required = True
        
        # Ajout à la mémoire
        memory.chat_memory.add_ai_message(final_response)
        
        # Construction de la réponse finale
        response_data = {
            "matched_bloc_response": final_response,
            "memory": memory.load_memory_variables({}).get("history", ""),
            "escalade_required": escalade_required,
            "escalade_type": priority_result.get("escalade_type", "admin"),
            "status": response_type,
            "priority_detected": priority_result.get("priority_detected", "NONE"),
            "processed_message": user_message,
            "response_length": len(final_response),
            "session_id": wa_id
        }
        
        logger.info(f"[{wa_id}] Response generated: type={response_type}, escalade={escalade_required}")
        
        return response_data
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        raise HTTPException(status_code=400, detail="Invalid JSON format")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/clear_memory/{wa_id}")
async def clear_memory(wa_id: str):
    """Efface la mémoire d'une conversation spécifique"""
    try:
        if wa_id in memory_store:
            del memory_store[wa_id]
            logger.info(f"Memory cleared for session: {wa_id}")
            return {"status": "success", "message": f"Memory cleared for {wa_id}"}
        else:
            return {"status": "info", "message": f"No memory found for {wa_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory for {wa_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_all_memory")
async def clear_all_memory():
    """Efface toute la mémoire"""
    try:
        global memory_store
        session_count = len(memory_store)
        memory_store.clear()
        logger.info(f"All memory cleared ({session_count} sessions)")
        return {"status": "success", "message": f"All memory cleared ({session_count} sessions)"}
    except Exception as e:
        logger.error(f"Error clearing all memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory_status")
async def memory_status():
    """Retourne le statut de la mémoire"""
    try:
        sessions = {}
        for wa_id, memory in memory_store.items():
            messages = memory.load_memory_variables({}).get("history", "")
            sessions[wa_id] = {
                "message_count": len(memory.chat_memory.messages),
                "last_interaction": "recent"  # Pourrait être enrichi avec timestamp
            }
        
        return {
            "active_sessions": len(memory_store),
            "sessions": sessions,
            "total_memory_size": sum(len(str(m.chat_memory.messages)) for m in memory_store.values())
        }
    except Exception as e:
        logger.error(f"Error getting memory status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return {
        "status": "healthy",
        "version": "4.0",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "active_sessions": len(memory_store)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)