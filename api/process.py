import os
from fastapi import FastAPI, HTTPException, Request
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import json

app = FastAPI()

# Récupération de la clé API depuis une variable d’environnement
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# Initialisation de la mémoire (stockée en mémoire temporaire, non persistante)
memory_store = {}

# Extraction des blocs de réponses à partir de la structure complexe
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
        "detection_paiement_formation": {...},  # Simplifié pour brièveté
        "style_obligatoire": {...},
        "gestion_agressivite": {
            "response": "Être impoli ne fera pas avancer la situation plus vite. Bien au contraire. Souhaites-tu que je te propose un poème ou une chanson d'amour pour apaiser ton cœur ? 💌"
        }
    },
    "blocs_reponses": {
        "bloc_A": {"id": "suivi_entreprise", "response": "Bien noté 👌\nPour te répondre au mieux, est-ce que tu sais :\n• Environ quand la formation s'est terminée ?\n• Et comment elle a été financée (par une entreprise directement, un OPCO, ou autre) ?\n👉 Une fois que j'ai ça, je te dis si le délai est normal ou si on doit faire une vérification ensemble 😊"},
        "bloc_B": {"id": "ambassadeur_nouveau", "response": "Salut 😄\nOui, on propose un programme super simple pour gagner de l'argent en nous envoyant des contacts intéressés 💼 💸\n\n🎯 Voici comment ça marche :\n✅ 1. Tu t'abonnes à nos réseaux pour suivre les actus\n👉 Insta : https://hi.switchy.io/InstagramWeiWei\n👉 Snap : https://hi.switchy.io/SnapChatWeiWei\n\n✅ 2. Tu nous transmets une liste de contacts\nNom + prénom + téléphone ou email (SIRET si c'est pro, c'est encore mieux)\n🔗 Formulaire : https://mrqz.to/AffiliationPromotion\n\n✅ 3. Tu touches une commission jusqu'à 60 % si un dossier est validé 🤑\nTu peux être payé directement sur ton compte perso (jusqu'à 3000 €/an et 3 virements)\nAu-delà, on peut t'aider à créer une micro-entreprise, c'est rapide ✨\n\nTu veux qu'on t'aide à démarrer ? Ou tu veux déjà envoyer ta première liste ? 📲"},
        # Ajoute les autres blocs (bloc_C, bloc_D, etc.) avec leurs "response"
    },
    "blocs_complementaires": {
        "sans_reseaux_sociaux": {"id": "sans_reseaux_sociaux", "response": "Pas de souci si tu n'es pas sur Insta ou Snap 😌 Tu peux simplement nous envoyer des contacts potentiellement intéressés. Ça fonctionne très bien aussi 😉"},
        "legalite_programme": {"id": "legalite_programme", "response": "On ne peut pas inscrire une personne dans une formation si son but est d'être rémunérée pour ça. En revanche, si tu fais la formation sérieusement, tu peux ensuite participer au programme d'affiliation et parrainer d'autres personnes."}
        # Ajoute les autres blocs complémentaires
    }
}

# Extraire tous les blocs avec une réponse
blocs = []
for category in ["blocs_reponses", "blocs_complementaires", "regles_comportementales"]:
    if category in blocs_data:
        for bloc_id, bloc_data in blocs_data[category].items():
            if "response" in bloc_data:
                blocs.append({"id": bloc_data.get("id", bloc_id), "response": bloc_data["response"]})

# Initialisation du vector store (lazy-loaded dans la fonction)
def initialize_vector_store():
    texts = [bloc["response"] for bloc in blocs]
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)
    return FAISS.from_documents(docs, embeddings)

vector_store = initialize_vector_store()

@app.post("/")
async def process_message(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message_original", "")
        matched_bloc_response = body.get("matched_bloc_response", "")
        wa_id = body.get("wa_id", "")

        # Initialisation de la mémoire pour cet utilisateur
        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory()
        memory = memory_store[wa_id]

        # Ajout du message utilisateur à la mémoire
        memory.chat_memory.add_user_message(user_message)

        # Si un bloc a déjà été trouvé, le retourner
        if matched_bloc_response:
            memory.chat_memory.add_ai_message(matched_bloc_response)
            return {
                "matched_bloc_response": matched_bloc_response,
                "memory": memory.load_memory_variables({})["history"]
            }

        # Recherche sémantique pour trouver un bloc pertinent
        similar_docs = vector_store.similarity_search(user_message, k=1)
        best_match = None
        for doc in similar_docs:
            for bloc in blocs:
                if bloc["response"] == doc.page_content:
                    best_match = bloc
                    break

        if best_match:
            matched_bloc_response = best_match["response"]
            memory.chat_memory.add_ai_message(matched_bloc_response)
            return {
                "matched_bloc_response": matched_bloc_response,
                "memory": memory.load_memory_variables({})["history"]
            }

        # Si rien n’est trouvé, escalade
        escalade_response = "Je vais faire suivre à la bonne personne dans l’équipe 😊 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner)."
        memory.chat_memory.add_ai_message(escalade_response)
        return {
            "matched_bloc_response": escalade_response,
            "memory": memory.load_memory_variables({})["history"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))