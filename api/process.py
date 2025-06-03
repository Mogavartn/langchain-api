import os
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import json

# Récupération de la clé API depuis une variable d’environnement
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialisation de la mémoire (stockée en mémoire temporaire, non persistante)
memory_store = {}

# Données des blocs (remplace par ton JSON complet)
blocs = [
    {"id": "legalite_programme", "response": "On ne peut pas inscrire une personne dans une formation si son but est d'être rémunérée pour ça. En revanche, si tu fais la formation sérieusement, tu peux ensuite participer au programme d'affiliation et parrainer d'autres personnes."},
    {"id": "sans_reseaux_sociaux", "response": "Pas de souci si tu n'es pas sur Insta ou Snap 😌 Tu peux simplement nous envoyer des contacts potentiellement intéressés. Ça fonctionne très bien aussi 😉"},
    {"id": "gestion_agressivite", "response": "Être impoli ne fera pas avancer la situation plus vite. Bien au contraire. Souhaites-tu que je te propose un poème ou une chanson d'amour pour apaiser ton cœur ? 💌"},
    # Ajoute tous tes blocs ici
]

# Création d'un vector store pour la recherche sémantique (initialisé une seule fois)
texts = [bloc["response"] for bloc in blocs]
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)
vector_store = FAISS.from_documents(docs, embeddings)

def handler(request):
    try:
        body = json.loads(request.body.decode('utf-8'))
        user_message = body.get("message_original", "")
        matched_bloc_response = body.get("matched_bloc_response", "")
        wa_id = body.get("wa_id", "")

        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory()
        memory = memory_store[wa_id]

        memory.chat_memory.add_user_message(user_message)

        if matched_bloc_response:
            memory.chat_memory.add_ai_message(matched_bloc_response)
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "matched_bloc_response": matched_bloc_response,
                    "memory": memory.load_memory_variables({})["history"]
                })
            }

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
                "statusCode": 200,
                "body": json.dumps({
                    "matched_bloc_response": matched_bloc_response,
                    "memory": memory.load_memory_variables({})["history"]
                })
            }

        escalade_response = "Je vais faire suivre à la bonne personne dans l’équipe 😊 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner)."
        memory.chat_memory.add_ai_message(escalade_response)
        return {
            "statusCode": 200,
            "body": json.dumps({
                "matched_bloc_response": escalade_response,
                "memory": memory.load_memory_variables({})["history"]
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }