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

app = FastAPI(title="JAK Company AI Agent API", version="8.2")

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

class MemoryManager:
    """Gestionnaire de mémoire optimisé pour limiter la taille"""

    @staticmethod
    def trim_memory(memory: ConversationBufferMemory, max_messages: int = 15):
        """Limite la mémoire aux N derniers messages pour économiser les tokens"""
        messages = memory.chat_memory.messages

        if len(messages) > max_messages:
            # Garder seulement les max_messages derniers
            memory.chat_memory.messages = messages[-max_messages:]
            logger.info(f"Memory trimmed to {max_messages} messages")

    @staticmethod
    def get_memory_summary(memory: ConversationBufferMemory) -> Dict[str, Any]:
        """Retourne un résumé de la mémoire"""
        messages = memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if hasattr(m, 'type') and m.type == 'human']),
            "ai_messages": len([m for m in messages if hasattr(m, 'type') and m.type == 'ai']),
            "memory_size_chars": sum(len(str(m.content)) for m in messages)
        }

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
    """Retourne le statut de la mémoire avec optimisations"""
    try:
        sessions = {}
        total_memory_chars = 0
        
        for wa_id, memory in memory_store.items():
            memory_summary = MemoryManager.get_memory_summary(memory)
            sessions[wa_id] = {
                **memory_summary,
                "last_interaction": "recent"  # Pourrait être enrichi avec timestamp
            }
            total_memory_chars += memory_summary["memory_size_chars"]
        
        return {
            "active_sessions": len(memory_store),
            "memory_type": "ConversationBufferMemory (Optimized)",
            "max_messages_per_session": 15,
            "sessions": sessions,
            "total_memory_size_chars": total_memory_chars,
            "optimization": "Auto-trim to 15 messages"
        }
    except Exception as e:
        logger.error(f"Error getting memory status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return {
        "status": "healthy",
        "version": "8.2",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "active_sessions": len(memory_store),
        "memory_type": "ConversationBufferMemory (Optimized)",
        "memory_optimization": "Auto-trim to 15 messages",
        "improvements": [
            "Fixed priority rules logic",
            "Eliminated code duplication",
            "Improved conversation context management",
            "Better CPF delay handling",
            "Enhanced payment context processing",
            "Corrected indentation issues",
            "Added ambassadeur type detection",
            "Improved ambassadeur explanation vs inscription logic",
            "Fixed ambassadeur context management - no more CPF confusion",
            "Enhanced memory debugging and robust pattern detection",
            "Extended confirmation patterns for ambassadeur context"
        ]
    }

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

class ConversationContextManager:
    """Gestionnaire du contexte conversationnel amélioré"""
    
    @staticmethod
    def analyze_conversation_context(user_message: str, memory: ConversationBufferMemory) -> Dict[str, Any]:
        """Analyse le contexte de la conversation pour adapter la réponse"""
        
        # Récupérer l'historique
        history = memory.chat_memory.messages
        message_count = len(history)
        
        # LOGS DEBUG MÉMOIRE
        logger.info(f"MÉMOIRE DEBUG: {message_count} messages dans l'historique")
        for i, msg in enumerate(history[-5:]):  # 5 derniers messages
            logger.info(f"Message {i}: Type={getattr(msg, 'type', 'unknown')}, Content={str(msg.content)[:100]}...")
        
        # Analyser si c'est un message de suivi
        follow_up_indicators = [
            "comment", "pourquoi", "vous pouvez", "tu peux", "aide", "démarrer",
            "oui", "ok", "d'accord", "et après", "ensuite", "comment faire",
            "vous pouvez m'aider", "tu peux m'aider", "comment ça marche",
            "ça marche comment", "pour les contacts"
        ]
        
        is_follow_up = any(indicator in user_message.lower() for indicator in follow_up_indicators)
        
        # Analyser le sujet précédent dans l'historique
        previous_topic = None
        last_bot_message = ""
        awaiting_cpf_info = False
        awaiting_financing_info = False
        awaiting_ambassadeur_info = False  # NOUVEAU CONTEXT
        
        if message_count > 0:
            # Chercher dans les derniers messages (jusqu'à 10 pour être sûr)
            for i, msg in enumerate(reversed(history[-10:])):  # Regarder les 10 derniers messages
                content = str(msg.content).lower()
                logger.info(f"ANALYSE MESSAGE {i}: {content[:150]}...")
                
                # DÉTECTION AMBASSADEUR AMÉLIORÉE - Patterns multiples
                is_ambassadeur_context = False
                
                # Pattern principal
                if "tu veux en savoir plus sur comment devenir ambassadeur" in content:
                    is_ambassadeur_context = True
                    logger.info("PATTERN 1 DÉTECTÉ: phrase complète ambassadeur")
                # Pattern alternatif 1
                elif "tu veux en savoir plus" in content and "ambassadeur" in content:
                    is_ambassadeur_context = True
                    logger.info("PATTERN 2 DÉTECTÉ: savoir plus + ambassadeur")
                # Pattern alternatif 2 - si le message contient ambassadeur ET une question
                elif "ambassadeur" in content and ("?" in content or "veux" in content):
                    is_ambassadeur_context = True
                    logger.info("PATTERN 3 DÉTECTÉ: ambassadeur + question")
                # Pattern alternatif 3 - détecter les blocs ambassadeur explication
                elif "partenaire de terrain" in content and "commission" in content:
                    is_ambassadeur_context = True
                    logger.info("PATTERN 4 DÉTECTÉ: bloc explication ambassadeur")
                # Pattern alternatif 4 - détecter "gagner de l'argent simplement"
                elif "gagner de l'argent simplement" in content:
                    is_ambassadeur_context = True
                    logger.info("PATTERN 5 DÉTECTÉ: gagner argent simplement")
                
                if is_ambassadeur_context:
                    awaiting_ambassadeur_info = True
                    last_bot_message = str(msg.content)
                    previous_topic = "ambassadeur_explication"
                    logger.info(f"✅ CONTEXTE AMBASSADEUR CONFIRMÉ dans: {content[:100]}...")
                    break  # IMPORTANT: sortir de la boucle pour éviter d'autres détections
                
                # Détecter si on attend des infos spécifiques (CPF/financement)
                if "comment la formation a été financée" in content:
                    awaiting_financing_info = True
                    last_bot_message = str(msg.content)
                    logger.info("CONTEXTE FINANCEMENT DÉTECTÉ")
                    
                if "environ quand la formation s'est terminée" in content:
                    awaiting_financing_info = True
                    last_bot_message = str(msg.content)
                    logger.info("CONTEXTE DATE FORMATION DÉTECTÉ")
                
                # Détecter le contexte CPF bloqué
                if "dossier cpf faisait partie des quelques cas bloqués" in content:
                    awaiting_cpf_info = True
                    last_bot_message = str(msg.content)
                    logger.info("CONTEXTE CPF BLOQUÉ DÉTECTÉ")
                
                # Détecter les sujets principaux (seulement si pas de contexte spécifique)
                if not awaiting_ambassadeur_info and not awaiting_cpf_info and not awaiting_financing_info:
                    if "ambassadeur" in content or "commission" in content:
                        previous_topic = "ambassadeur"
                    elif "paiement" in content or "formation" in content:
                        previous_topic = "paiement"
                    elif "cpf" in content:
                        previous_topic = "cpf"
        
        # Logging final du contexte
        logger.info(f"CONTEXTE FINAL: awaiting_ambassadeur={awaiting_ambassadeur_info}, awaiting_cpf={awaiting_cpf_info}, awaiting_financing={awaiting_financing_info}, topic={previous_topic}")
        
        return {
            "message_count": message_count,
            "is_follow_up": is_follow_up,
            "previous_topic": previous_topic,
            "needs_greeting": message_count == 0,
            "conversation_flow": "continuing" if message_count > 0 else "starting",
            "awaiting_cpf_info": awaiting_cpf_info,
            "awaiting_financing_info": awaiting_financing_info,
            "awaiting_ambassadeur_info": awaiting_ambassadeur_info,  # NOUVEAU
            "last_bot_message": last_bot_message
        }

class PaymentContextProcessor:
    """Processeur spécialisé pour le contexte paiement formation"""
    
    @staticmethod
    def extract_financing_type(message: str) -> Optional[str]:
        """Extrait le type de financement du message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['cpf', 'compte personnel']):
            return 'CPF'
        elif any(word in message_lower for word in ['opco', 'opérateur']):
            return 'OPCO'
        elif any(word in message_lower for word in ['direct', 'entreprise', 'particulier']):
            return 'direct'
        
        return None
    
    @staticmethod
    def extract_time_delay(message: str) -> Optional[int]:
        """Extrait le délai en mois du message"""
        message_lower = message.lower()
        
        # Patterns pour extraire les délais
        patterns = [
            r'(\d+)\s*mois',
            r'il y a\s*(\d+)\s*mois',
            r'ça fait\s*(\d+)\s*mois',
            r'depuis\s*(\d+)\s*mois',
            r'(\d+)\s*mois que'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    @staticmethod
    def handle_cpf_delay_context(delay_months: int, user_message: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Gère le contexte spécifique CPF avec délai"""
        
        if delay_months >= 2:  # CPF délai dépassé
            # Vérifier si c'est une réponse à la question de blocage CPF
            if conversation_context.get("awaiting_cpf_info"):
                user_lower = user_message.lower()
                
                # Si l'utilisateur confirme qu'il était informé du blocage
                if any(word in user_lower for word in ['oui', 'yes', 'informé', 'dit', 'déjà', 'je sais']):
                    return {
                        "use_matched_bloc": False,
                        "priority_detected": "CPF_BLOQUE_CONFIRME",
                        "response": """On comprend parfaitement ta frustration. Ce dossier fait partie des quelques cas (moins de 50 sur plus de 2500) bloqués depuis la réforme CPF de février 2025. Même nous n'avons pas été payés. Le blocage est purement administratif, et les délais sont impossibles à prévoir. On te tiendra informé dès qu'on a du nouveau. Inutile de relancer entre-temps 🙏

Tous les éléments nécessaires ont bien été transmis à l'organisme de contrôle 📋🔍
Mais le problème, c'est que la Caisse des Dépôts demande des documents que le centre de formation envoie sous une semaine…
Et ensuite, ils prennent parfois jusqu'à 2 mois pour demander un nouveau document, sans donner de réponse entre-temps.

✅ On accompagne au maximum le centre de formation pour que tout rentre dans l'ordre.
⚠️ On est aussi impactés financièrement : chaque formation a un coût pour nous.
🤞 On garde confiance et on espère une issue favorable.
🗣️ Et surtout, on s'engage à revenir vers chaque personne concernée dès qu'on a du nouveau.""",
                        "context": conversation_context,
                        "escalade_type": None
                    }
                else:
                    # Escalade pour vérification
                    return {
                        "use_matched_bloc": False,
                        "priority_detected": "CPF_VERIFICATION_ESCALADE",
                        "response": """Parfait, je vais faire suivre ta demande à notre équipe ! 😊

🕐 Notre équipe est disponible du lundi au vendredi, de 9h à 17h. On te tiendra informé dès que possible ✅

🔄 Escalade: AGENT ADMIN""",
                        "context": conversation_context,
                        "escalade_type": "admin"
                    }
            else:
                # Première fois qu'on détecte un délai CPF dépassé
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "CPF_DELAI_DEPASSE_FILTRAGE",
                    "response": """Juste avant que je transmette ta demande 🙏

Est-ce que tu as déjà été informé par l'équipe que ton dossier CPF faisait partie des quelques cas bloqués par la Caisse des Dépôts ?

👉 Si oui, je te donne directement toutes les infos liées à ce blocage.
Sinon, je fais remonter ta demande à notre équipe pour vérification ✅""",
                    "context": conversation_context,
                    "awaiting_cpf_info": True
                }
        
        return None

class MessageProcessor:
    """Classe principale pour traiter les messages avec contexte"""

    @staticmethod
    def is_aggressive(message: str) -> bool:
        """Détecte l'agressivité en évitant les faux positifs"""
        
        message_lower = message.lower()
        
        # Liste des mots agressifs avec leurs contextes d'exclusion
        aggressive_patterns = [
            ("merde", []), # Pas d'exclusion
            ("nul", ["nul part", "nulle part"]), # Exclure "nul part"
            ("énervez", []),
            ("batards", []),
            ("putain", []),
            ("chier", [])
        ]
        
        # Vérification spéciale pour "con" - doit être un mot isolé
        if " con " in f" {message_lower} " or message_lower.startswith("con ") or message_lower.endswith(" con"):
            # Exclure les mots contenant "con" comme "contacts", "conseil", "condition", etc.
            exclusions = [
                "contacts", "contact", "conseil", "conseils", "condition", "conditions",
                "concernant", "concerne", "construction", "consultation", "considère",
                "consommation", "consommer", "constitue", "contenu", "contexte",
                "contrôle", "contraire", "confiance", "confirmation", "conformité"
            ]
            
            # Vérifier qu'il n'y a pas ces mots dans le message
            if not any(exclusion in message_lower for exclusion in exclusions):
                return True
        
        # Vérifier les autres mots agressifs
        for aggressive_word, exclusions in aggressive_patterns:
            if aggressive_word in message_lower:
                # Vérifier que ce n'est pas dans un contexte d'exclusion
                if not any(exclusion in message_lower for exclusion in exclusions):
                    return True
        
        return False

    @staticmethod
    def detect_ambassadeur_type(message: str) -> Optional[str]:
        """Détecte le type de demande ambassadeur"""
        message_lower = message.lower()
        
        # Questions d'explication (priorité haute)
        explanation_keywords = [
            "c'est quoi", "qu'est-ce que", "que fait", "kesako", 
            "définition", "expliquer", "rôle", "role"
        ]
        
        # Vérifier si c'est une question d'explication
        if any(keyword in message_lower for keyword in explanation_keywords) and "ambassadeur" in message_lower:
            return "explication"
        
        if "ambassadeur" in message_lower and "?" in message_lower:
            return "explication"
        
        # Demandes d'inscription/action
        action_keywords = [
            "je veux", "j'aimerais", "comment devenir", "comment faire", 
            "inscription", "m'inscrire", "rejoindre", "participer",
            "gagner argent", "faire argent", "commission"
        ]
        
        if any(keyword in message_lower for keyword in action_keywords):
            return "inscription"
        
        return "inscription"  # Par défaut

    @staticmethod
    def detect_priority_rules(user_message: str, matched_bloc_response: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les règles de priorité avec prise en compte du contexte - VERSION CORRIGÉE V8.2"""
        
        message_lower = user_message.lower()
        
        # ÉTAPE 0: NOUVELLE - Traitement du contexte ambassadeur en attente (PRIORITÉ ABSOLUE)
        if conversation_context.get("awaiting_ambassadeur_info"):
            logger.info(f"🎯 TRAITEMENT CONTEXTE AMBASSADEUR: message='{user_message}'")
            
            # Patterns de confirmation étendus
            confirmation_patterns = [
                'oui', 'yes', 'ok', 'd\'accord', 'exact', 'je veux', 'intéresse', 
                'ça m\'intéresse', 'ca m\'interesse', 'je veux savoir', 'oui je veux',
                'oui je veux savoir', 'bien sûr', 'bien sur', 'parfait', 'super',
                'oui ça m\'intéresse', 'oui ca m\'interesse', 'pourquoi pas',
                'je suis intéressé', 'je suis interessé', 'allons-y', 'vas-y'
            ]
            
            # Patterns de refus étendus  
            refusal_patterns = [
                'non', 'no', 'pas intéressé', 'pas interessé', 'jamais', 'pas pour moi',
                'non merci', 'ça ne m\'intéresse pas', 'ca ne m\'interesse pas',
                'pas maintenant', 'une autre fois', 'plus tard'
            ]
            
            # Vérification si c'est une confirmation
            is_confirmation = any(pattern in message_lower for pattern in confirmation_patterns)
            is_refusal = any(pattern in message_lower for pattern in refusal_patterns)
            
            logger.info(f"ANALYSE RÉPONSE: is_confirmation={is_confirmation}, is_refusal={is_refusal}")
            
            # Si l'utilisateur confirme qu'il veut en savoir plus
            if is_confirmation:
                logger.info("✅ CONFIRMATION AMBASSADEUR DÉTECTÉE")
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "AMBASSADEUR_SUITE_INSCRIPTION",
                    "response": """Parfait ! 😄 

Je vais t'expliquer comment démarrer :

✅ Étape 1 : Tu t'abonnes à nos réseaux
📱 Insta : https://hi.switchy.io/InstagramWeiWei
📱 Snap : https://hi.switchy.io/SnapChatWeiWei

✅ Étape 2 : Tu nous envoies une liste de contacts intéressés (nom, prénom, téléphone ou email).
➕ Si c'est une entreprise ou un pro, le SIRET est un petit bonus 😊
🔗 Formulaire ici : https://mrqz.to/AffiliationPromotion

✅ Étape 3 : Si un dossier est validé, tu touches une commission jusqu'à 60 % 💰
Et tu peux même être payé sur ton compte perso (jusqu'à 3000 €/an et 3 virements)

Tu veux qu'on t'aide à démarrer ou tu as des questions ? 📝""",
                    "context": conversation_context,
                    "escalade_type": None
                }
            
            # Si l'utilisateur refuse
            elif is_refusal:
                logger.info("❌ REFUS AMBASSADEUR DÉTECTÉ")
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "AMBASSADEUR_PAS_INTERESSE",
                    "response": """Pas de souci ! 😊

Si tu changes d'avis ou si tu as d'autres questions, n'hésite pas à revenir vers nous.

Y a-t-il autre chose sur quoi je peux t'aider ? 👍""",
                    "context": conversation_context,
                    "escalade_type": None
                }
            
            # Si c'est une autre question, continuer avec la logique normale
            else:
                logger.info("🔄 AUTRE QUESTION DANS CONTEXTE AMBASSADEUR - RESET")
                # Reset du contexte ambassadeur pour traiter la nouvelle question
                conversation_context["awaiting_ambassadeur_info"] = False
        
        # ÉTAPE 1: Traitement des réponses aux questions spécifiques en cours
        if conversation_context.get("awaiting_financing_info"):
            financing_type = PaymentContextProcessor.extract_financing_type(user_message)
            delay_months = PaymentContextProcessor.extract_time_delay(user_message)
            
            if financing_type == "CPF" and delay_months:
                cpf_result = PaymentContextProcessor.handle_cpf_delay_context(
                    delay_months, user_message, conversation_context
                )
                if cpf_result:
                    return cpf_result
            
            elif financing_type == "OPCO" and delay_months and delay_months >= 2:
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "OPCO_DELAI_DEPASSE",
                    "response": """Merci pour ta réponse 🙏

Pour un financement via un OPCO, le délai moyen est de 2 mois. Certains dossiers peuvent aller jusqu'à 6 mois ⏳

Mais vu que cela fait plus de 2 mois, on préfère ne pas te faire attendre plus longtemps sans retour.

👉 Je vais transmettre ta demande à notre équipe pour qu'on vérifie ton dossier dès maintenant 📋

🔄 ESCALADE AGENT ADMIN

🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).
On te tiendra informé dès qu'on a une réponse ✅""",
                    "context": conversation_context,
                    "escalade_type": "admin"
                }
            
            elif financing_type and not delay_months:
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "DEMANDE_DATE_FORMATION",
                    "response": "Et environ quand la formation s'est-elle terminée ?",
                    "context": conversation_context,
                    "awaiting_financing_info": True
                }
        
        # ÉTAPE 2: Traitement du contexte CPF bloqué (avec protection contre contexte ambassadeur)
        if conversation_context.get("awaiting_cpf_info") and not conversation_context.get("awaiting_ambassadeur_info"):
            return PaymentContextProcessor.handle_cpf_delay_context(0, user_message, conversation_context)
        
        # ÉTAPE 3: Agressivité (priorité haute pour couper court)
        if MessageProcessor.is_aggressive(user_message):
            return {
                "use_matched_bloc": False,
                "priority_detected": "AGRESSIVITE",
                "response": "Être impoli ne fera pas avancer la situation plus vite. Bien au contraire. Souhaites-tu que je te propose un poème ou une chanson d'amour pour apaiser ton cœur ? 💌",
                "context": conversation_context
            }
        
        # ÉTAPE 4: Gestion spécifique AMBASSADEUR
        if "ambassadeur" in message_lower:
            ambassadeur_type = MessageProcessor.detect_ambassadeur_type(user_message)
            
            # Si c'est une question d'explication et qu'on a le bon bloc
            if ambassadeur_type == "explication" and matched_bloc_response and "ambassadeur" in matched_bloc_response.lower():
                # Vérifier si c'est le bon bloc d'explication
                if "partenaire de terrain" in matched_bloc_response or "gagner de l'argent simplement" in matched_bloc_response:
                    return {
                        "use_matched_bloc": True,
                        "priority_detected": "AMBASSADEUR_EXPLICATION",
                        "response": matched_bloc_response,
                        "context": conversation_context
                    }
            
            # Si c'est une demande d'inscription et qu'on a le bon bloc
            elif ambassadeur_type == "inscription" and matched_bloc_response and "ambassadeur" in matched_bloc_response.lower():
                # Vérifier si c'est le bloc d'inscription (étapes 1-2-3)
                if "Étape 1" in matched_bloc_response or "t'abonnes à nos réseaux" in matched_bloc_response:
                    return {
                        "use_matched_bloc": True,
                        "priority_detected": "AMBASSADEUR_INSCRIPTION", 
                        "response": matched_bloc_response,
                        "context": conversation_context
                    }
        
        # ÉTAPE 5: Détection problème paiement formation
        payment_keywords = [
            "pas été payé", "rien reçu", "virement", "attends",
            "paiement", "argent", "retard", "promesse", "veux être payé",
            "payé pour ma formation", "être payé pour"
        ]
        
        if any(keyword in message_lower for keyword in payment_keywords):
            # Si c'est un message de suivi sur le paiement
            if conversation_context["message_count"] > 0 and conversation_context["is_follow_up"]:
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "PAIEMENT_SUIVI",
                    "response": None,  # Laisser l'IA gérer avec contexte
                    "context": conversation_context,
                    "use_ai": True
                }
            # Si un bloc est matché pour le paiement, l'utiliser
            elif matched_bloc_response and ("paiement" in matched_bloc_response.lower() or "délai" in matched_bloc_response.lower()):
                return {
                    "use_matched_bloc": True,
                    "priority_detected": "PAIEMENT_FORMATION_BLOC",
                    "response": matched_bloc_response,
                    "context": conversation_context
                }
            # Sinon, fallback paiement
            else:
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "PAIEMENT_SANS_BLOC",
                    "response": """Salut 👋

Je comprends que tu aies des questions sur le paiement 💰

Je vais faire suivre ta demande à notre équipe spécialisée qui te recontactera rapidement ✅

🕐 Horaires : Lundi-Vendredi, 9h-17h""",
                    "context": conversation_context,
                    "escalade_type": "admin"
                }
        
        # ÉTAPE 6: Bloc matché (si pas de problème de paiement détecté)
        if matched_bloc_response and matched_bloc_response.strip():
            # Éviter les répétitions si c'est un message de suivi
            if conversation_context["is_follow_up"] and conversation_context["message_count"] > 0:
                return {
                    "use_matched_bloc": False,
                    "priority_detected": "FOLLOW_UP_VS_BLOC",
                    "response": None,  # Laisser l'IA gérer
                    "context": conversation_context,
                    "use_ai": True
                }
            else:
                return {
                    "use_matched_bloc": True,
                    "priority_detected": "BLOC_MATCHE",
                    "response": matched_bloc_response,
                    "context": conversation_context
                }
        
        # ÉTAPE 7: Messages de suivi généraux
        if conversation_context["is_follow_up"] and conversation_context["message_count"] > 0:
            return {
                "use_matched_bloc": False,
                "priority_detected": "FOLLOW_UP_CONVERSATION",
                "response": None,  # Laisser l'IA gérer
                "context": conversation_context,
                "use_ai": True
            }
        
        # ÉTAPE 8: Escalade automatique
        escalade_type = ResponseValidator.validate_escalade_keywords(user_message)
        if escalade_type:
            return {
                "use_matched_bloc": False,
                "priority_detected": "ESCALADE_AUTO",
                "escalade_type": escalade_type,
                "response": "🔄 ESCALADE AGENT ADMIN\n\n🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\n👉 On te tiendra informé dès qu'on a du nouveau ✅",
                "context": conversation_context
            }
        
        # ÉTAPE 9: Fallback général
        return {
            "use_matched_bloc": False,
            "priority_detected": "FALLBACK_GENERAL",
            "context": conversation_context,
            "response": None,
            "use_ai": True
        }

@app.post("/")
async def process_message(request: Request):
    """Point d'entrée principal pour traiter les messages avec contexte"""
    try:
        # Gestion robuste du parsing JSON
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            # Si le JSON est mal formé, tenter de lire le contenu brut
            raw_body = await request.body()
            logger.error(f"JSON decode error: {str(e)}, raw body: {raw_body.decode('utf-8')[:500]}")
            
            # Essayer de nettoyer et re-parser
            try:
                clean_body = raw_body.decode('utf-8').strip()
                body = json.loads(clean_body)
            except:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        
        # Logging amélioré pour debug
        logger.info(f"Received body type: {type(body)}")
        logger.info(f"Body keys: {list(body.keys()) if isinstance(body, dict) else 'Not a dict'}")
        
        # Extraction des données avec fallbacks AMÉLIORÉE
        if isinstance(body, dict):
            user_message = body.get("message_original", body.get("message", ""))
            matched_bloc_response = body.get("matched_bloc_response", "")
            wa_id = body.get("wa_id", "default_wa_id")
        else:
            # Si ce n'est pas un dict, essayer de traiter comme string
            user_message = str(body) if body else ""
            matched_bloc_response = ""
            wa_id = "fallback_wa_id"
        
        logger.info(f"[{wa_id}] Processing: message='{user_message[:50]}...', has_bloc={bool(matched_bloc_response)}")
        
        # Validation des entrées
        if not user_message or not user_message.strip():
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Nettoyage des données
        user_message = ResponseValidator.clean_response(user_message)
        matched_bloc_response = ResponseValidator.clean_response(matched_bloc_response)
        
        # Gestion de la mémoire conversation
        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        
        memory = memory_store[wa_id]
        
        # Optimiser la mémoire en limitant la taille
        MemoryManager.trim_memory(memory, max_messages=15)
        
        # Analyser le contexte de conversation avec le nouveau manager
        conversation_context = ConversationContextManager.analyze_conversation_context(user_message, memory)
        
        # Résumé mémoire pour logs
        memory_summary = MemoryManager.get_memory_summary(memory)
        
        logger.info(f"[{wa_id}] Conversation context: {conversation_context}")
        logger.info(f"[{wa_id}] Memory summary: {memory_summary}")
        
        # Ajouter le message utilisateur à la mémoire
        memory.chat_memory.add_user_message(user_message)
        
        # Application des règles de priorité avec contexte
        priority_result = MessageProcessor.detect_priority_rules(
            user_message,
            matched_bloc_response,
            conversation_context
        )
        
        # Construction de la réponse selon la priorité et le contexte
        final_response = None
        response_type = "unknown"
        escalade_required = False
        
        if priority_result.get("use_matched_bloc") and priority_result.get("response"):
            final_response = priority_result["response"]
            response_type = "exact_match_enforced"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "CPF_BLOQUE_CONFIRME":
            final_response = priority_result["response"]
            response_type = "cpf_blocked_confirmed"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "CPF_DELAI_DEPASSE_FILTRAGE":
            final_response = priority_result["response"]
            response_type = "cpf_delay_filtering"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "OPCO_DELAI_DEPASSE":
            final_response = priority_result["response"]
            response_type = "opco_delay_exceeded"
            escalade_required = True
            
        elif priority_result.get("priority_detected") == "DEMANDE_DATE_FORMATION":
            final_response = priority_result["response"]
            response_type = "asking_formation_date"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "AGRESSIVITE":
            final_response = priority_result["response"]
            response_type = "agressivite_detected"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "AMBASSADEUR_EXPLICATION":
            final_response = priority_result["response"]
            response_type = "ambassadeur_explication"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "AMBASSADEUR_INSCRIPTION":
            final_response = priority_result["response"]
            response_type = "ambassadeur_inscription"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "AMBASSADEUR_SUITE_INSCRIPTION":
            final_response = priority_result["response"]
            response_type = "ambassadeur_suite_inscription"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "AMBASSADEUR_PAS_INTERESSE":
            final_response = priority_result["response"]
            response_type = "ambassadeur_pas_interesse"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "FOLLOW_UP_CONVERSATION":
            final_response = None # Sera géré par l'IA
            response_type = "follow_up_ai_handled"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "PAIEMENT_SUIVI":
            final_response = None # Sera géré par l'IA
            response_type = "paiement_suivi_ai_handled"
            escalade_required = False
            
        elif priority_result.get("priority_detected") == "ESCALADE_AUTO":
            final_response = priority_result["response"]
            response_type = "auto_escalade"
            escalade_required = True
            
        elif priority_result.get("priority_detected") == "PAIEMENT_SANS_BLOC":
            final_response = priority_result["response"]
            response_type = "paiement_fallback"
            escalade_required = True
            
        elif priority_result.get("priority_detected") == "BLOC_MATCHE":
            final_response = priority_result["response"]
            response_type = "bloc_matched"
            escalade_required = False
            
        else:
            # Utiliser l'IA pour une réponse contextuelle ou fallback
            final_response = None
            response_type = "ai_contextual_response"
            escalade_required = priority_result.get("use_ai", False)
        
        # Si pas de réponse finale, utiliser un fallback
        if final_response is None:
            # Adapter le fallback selon le contexte
            if conversation_context["needs_greeting"]:
                final_response = """Salut 👋

Je vais faire suivre ta demande à notre équipe pour qu'elle puisse t'aider au mieux 😊

🕐 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner).
On te tiendra informé dès que possible ✅

En attendant, peux-tu me préciser un peu plus ce que tu recherches ?"""
            else:
                final_response = """Parfait, je vais faire suivre ta demande à notre équipe ! 😊

🕐 Notre équipe est disponible du lundi au vendredi, de 9h à 17h.
On te tiendra informé dès que possible ✅"""
            
            response_type = "fallback_with_context"
            escalade_required = True
        
        # Ajout à la mémoire seulement si on a une réponse finale
        if final_response:
            memory.chat_memory.add_ai_message(final_response)
        
        # Optimiser la mémoire après ajout
        MemoryManager.trim_memory(memory, max_messages=15)
        
        # Construction de la réponse finale avec contexte
        response_data = {
            "matched_bloc_response": final_response,
            "memory": memory.load_memory_variables({}).get("history", ""),
            "escalade_required": escalade_required,
            "escalade_type": priority_result.get("escalade_type", "admin"),
            "status": response_type,
            "priority_detected": priority_result.get("priority_detected", "NONE"),
            "processed_message": user_message,
            "response_length": len(final_response) if final_response else 0,
            "session_id": wa_id,
            "conversation_context": conversation_context,
            "memory_summary": memory_summary
        }
        
        logger.info(f"[{wa_id}] Response generated: type={response_type}, escalade={escalade_required}, memory={memory_summary}")
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        
        # Retourner une réponse de fallback au lieu d'une erreur
        return {
            "matched_bloc_response": "Salut 👋\n\nJe rencontre un petit problème technique. Notre équipe va regarder ça et te recontacter rapidement ! 😊\n\n🕐 Horaires : Lundi-Vendredi, 9h-17h",
            "memory": "",
            "escalade_required": True,
            "escalade_type": "technique",
            "status": "error_fallback",
            "priority_detected": "ERROR",
            "processed_message": "error_occurred",
            "response_length": 150,
            "session_id": "error_session",
            "conversation_context": {"message_count": 0, "is_follow_up": False, "needs_greeting": True},
            "memory_summary": {"total_messages": 0, "user_messages": 0, "ai_messages": 0, "memory_size_chars": 0}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)