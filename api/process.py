import os
from fastapi import FastAPI, HTTPException, Request
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Récupération de la clé API depuis une variable d’environnement
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# Initialisation de la mémoire (stockée en mémoire temporaire, non persistante)
memory_store = {}

# Base de données complète extraite du JSON (inchangée)
blocs_data = {
    "meta": {
        "version": "8.0",
        "description": "Base de données complète IA WhatsApp - JAK Company x WeiWei",
        "priority_rules": [
            "DETECTION_PROBLEME_PAIEMENT_FORMATION",
            "FILTRAGE_OBLIGATOIRE_AVANT_ESCALADE",
            "REPRODUCTION_EXACTE_DES_BLOCS",
            "TON_WHATSAPP_AVEC_EMOJIS"
        ]
    },
    "config": {
        "horaires_support": "Du lundi au vendredi, de 9h à 17h (hors pause déjeuner)",
        "reseaux_sociaux": {
            "instagram": "https://hi.switchy.io/InstagramWeiWei",
            "snapchat": "https://hi.switchy.io/SnapChatWeiWei"
        },
        "formulaire_affiliation": "https://mrqz.to/AffiliationPromotion",
        "programme_jak": "https://swiy.co/programmeprojak"
    },
    "regles_comportementales": {
        "detection_paiement_formation": {
            "priorite": "ABSOLUE",
            "exemples_intentions": [
                "j'ai pas été payé",
                "toujours rien reçu",
                "je devais avoir un virement",
                "on m'a dit que j'allais être payé",
                "ça fait 3 mois j'attends",
                "une attente d'argent",
                "plainte sur non-versement",
                "retard de paiement",
                "promesse non tenue"
            ],
            "questions_obligatoires": [
                "Comment la formation a-t-elle été financée ? (CPF, OPCO, ou paiement direct)",
                "Et environ quand la formation s'est-elle terminée ?"
            ],
            "interdictions": [
                "Ne jamais répondre de façon vague",
                "Ne jamais parler de 'service', 'dossier', 'période' sans précision",
                "Ne jamais proposer action extérieure",
                "Ne jamais escalader sans les 2 infos obligatoires"
            ]
        },
        "style_obligatoire": {
            "ton": "chaleureux et humain",
            "emojis": "naturels selon contexte",
            "langage": "oral, fluide, sans administratif",
            "structure": "paragraphes courts ou étapes claires",
            "reproduction": "blocs mot pour mot, sans résumé ni combinaison"
        },
        "gestion_agressivite": {
            "response": "Être impoli ne fera pas avancer la situation plus vite. Bien au contraire. Souhaites-tu que je te propose un poème ou une chanson d'amour pour apaiser ton cœur ? 💌"
        }
    },
    "blocs_reponses": {
        "bloc_F": {
            "id": "paiement_formation",
            "response": "Salut 👋\nLe délai dépend du type de formation qui va être rémunérée et surtout de la manière dont elle a été financée 💡\n\n🔹 Si la formation a été payée directement (par un particulier ou une entreprise)\n→ Le paiement est effectué sous 7 jours après la fin de la formation et réception du dossier complet (émargement + administratif) 🧾\n\n🔹 Si la formation a été financée par le CPF\n→ Le paiement se fait à partir de 45 jours, mais uniquement à compter de la réception effective des feuilles d'émargement signées ✍\n\n🔹 Si le dossier est financé via un OPCO\n→ Le délai moyen est de 2 mois, mais certains dossiers prennent jusqu'à 6 mois ⏳\n\nPour que je puisse t'aider au mieux, est-ce que tu peux me préciser :\n• Comment la formation a été financée ?\n• Et environ quand elle s'est terminée ?\n\n👉 Je te dirai si c'est encore dans les délais normaux, ou si on fait une vérification 😊"
        }
        # Autres blocs inchangés pour brièveté
    },
    "blocs_escalade": {
        "escalade_agent_admin": {
            "id": "escalade_agent_admin",
            "declencheurs": [
                "preuve de virement",
                "fichier ou échéance",
                "retard anormal",
                "vérification dossier"
            ],
            "response": "🔁 ESCALADE AGENT ADMIN\n\n📅 Rappel :\n\"Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nNous te répondrons dès que possible.\"\n\n🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nOn te tiendra informé dès qu'on a du nouveau ✅"
        }
    }
    # Autres blocs inchangés pour brièveté
}

# Extraire tous les blocs avec une réponse
blocs = []
for category in ["blocs_reponses", "blocs_escalade", "blocs_complementaires", "regles_comportementales"]:
    if category in blocs_data:
        for bloc_id, bloc_data in blocs_data[category].items():
            if "response" in bloc_data:
                blocs.append({
                    "id": bloc_data.get("id", bloc_id),
                    "response": bloc_data["response"],
                    "category": category
                })

# Initialisation du vector store
def initialize_vector_store():
    texts = [bloc["response"] for bloc in blocs]
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)
    return FAISS.from_documents(docs, embeddings)

vector_store = initialize_vector_store()

# Détection des problèmes de paiement
def detect_payment_issue(message):
    payment_keywords = [
        "c’est quand que je vais recevoir mon paiement", "j'ai pas été payé", "toujours rien reçu",
        "je devais avoir un virement", "on m'a dit que j'allais être payé"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in payment_keywords)

# Détection des déclencheurs d'escalade
def detect_escalade_trigger(message):
    escalade_triggers = blocs_data["blocs_escalade"]["escalade_agent_admin"]["declencheurs"]
    message_lower = message.lower()
    return any(trigger in message_lower for trigger in escalade_triggers)

@app.post("/")
async def process_message(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message_original", body.get("message", ""))
        matched_bloc_response = body.get("matched_bloc_response", "")
        wa_id = body.get("wa_id", "default_wa_id")

        logger.info(f"Raw request body: {body}")
        logger.info(f"Processing message: {user_message}, wa_id: {wa_id}, matched_bloc_response: {matched_bloc_response}")

        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory()
            logger.info(f"New memory initialized for wa_id: {wa_id}")
        memory = memory_store[wa_id]

        if detect_payment_issue(user_message) or detect_escalade_trigger(user_message):
            memory.clear()
            logger.info(f"Memory cleared for wa_id: {wa_id}")

        memory.chat_memory.add_user_message(user_message)

        if matched_bloc_response and matched_bloc_response.strip():
            logger.info(f"Using pre-matched response from Fuzzy Matcher: {matched_bloc_response}")
            memory.chat_memory.add_ai_message(matched_bloc_response)
            return {
                "matched_bloc_response": matched_bloc_response,
                "memory": memory.load_memory_variables({})["history"],
                "escalade_required": False,
                "use_exact_match": True  # Indicateur pour forcer l'utilisation exacte
            }

        if detect_escalade_trigger(user_message):
            escalade_bloc = next((bloc for bloc in blocs if bloc["id"] == "escalade_agent_admin"), None)
            if escalade_bloc:
                memory.chat_memory.add_ai_message(escalade_bloc["response"])
                return {
                    "matched_bloc_response": escalade_bloc["response"],
                    "memory": memory.load_memory_variables({})["history"],
                    "escalade_required": True,
                    "escalade_type": "admin"
                }

        if detect_payment_issue(user_message):
            payment_bloc = next((bloc for bloc in blocs if bloc["id"] == "paiement_formation"), None)
            if payment_bloc:
                memory.chat_memory.add_ai_message(payment_bloc["response"])
                return {
                    "matched_bloc_response": payment_bloc["response"],
                    "memory": memory.load_memory_variables({})["history"],
                    "priority_detection": "PAYMENT_ISSUE"
                }

        escalade_response = "Je vais faire suivre à la bonne personne dans l'équipe 😊 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner)."
        memory.chat_memory.add_ai_message(escalade_response)
        return {
            "matched_bloc_response": escalade_response,
            "memory": memory.load_memory_variables({})["history"],
            "escalade_required": True,
            "escalade_type": "default"
        }

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints inchangés pour brièveté
@app.post("/clear_memory")
async def clear_memory(request: Request):
    try:
        body = await request.json()
        wa_id = body.get("wa_id", "")
        if wa_id in memory_store:
            del memory_store[wa_id]
        return {"status": "success", "message": f"Memory cleared for wa_id: {wa_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_all_memory")
async def clear_all_memory():
    try:
        global memory_store
        memory_store.clear()
        return {"status": "success", "message": "All conversation memory cleared"}
    except Exception as e:
        logger.error(f"Error clearing all memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)