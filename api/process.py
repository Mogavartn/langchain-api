import os
from fastapi import FastAPI, HTTPException, Request
from langchain.memory import ConversationBufferMemory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

memory_store = {}

@app.post("/")
async def process_message(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message_original", body.get("message", ""))
        matched_bloc_response = body.get("matched_bloc_response", "")
        wa_id = body.get("wa_id", "default_wa_id")

        logger.info(f"Raw request body: {body}")
        logger.info(f"Processing: message={user_message}, wa_id={wa_id}, matched_bloc={matched_bloc_response}")

        # Ignorer la mémoire si matched_bloc_response est présent
        if matched_bloc_response and matched_bloc_response.strip():
            logger.info(f"Using matched_bloc_response, ignoring memory: {matched_bloc_response}")
            return {
                "matched_bloc_response": matched_bloc_response,
                "memory": "",  # Retourner une mémoire vide
                "escalade_required": False,
                "use_exact_match": True,
                "status": "exact_match_enforced"
            }

        if wa_id not in memory_store:
            memory_store[wa_id] = ConversationBufferMemory()
        memory = memory_store[wa_id]

        memory.chat_memory.add_user_message(user_message)

        if "retard anormal" in user_message.lower():
            escalade_response = "🔁 ESCALADE AGENT ADMIN\n\n📅 Rappel : \"Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\"\n🕐 On te tiendra informé dès qu'on a du nouveau ✅"
            memory.chat_memory.add_ai_message(escalade_response)
            return {
                "matched_bloc_response": escalade_response,
                "memory": memory.load_memory_variables({})["history"],
                "escalade_required": True,
                "escalade_type": "admin"
            }

        if "paiement" in user_message.lower():
            payment_response = "Salut 👋\nLe délai dépend du type de formation qui va être rémunérée et surtout de la manière dont elle a été financée 💡\n\n🔹 Si la formation a été payée directement...\n→ Le paiement est effectué sous 7 jours...\n👉 Je te dirai si c'est dans les délais 😊"
            memory.chat_memory.add_ai_message(payment_response)
            return {
                "matched_bloc_response": payment_response,
                "memory": memory.load_memory_variables({})["history"],
                "priority_detection": "PAYMENT_ISSUE"
            }

        escalade_response = "Je vais faire suivre à la bonne personne dans l'équipe 😊 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner)."
        memory.chat_memory.add_ai_message(escalade_response)
        logger.warning("Falling back to default escalade due to no match")
        return {
            "matched_bloc_response": escalade_response,
            "memory": memory.load_memory_variables({})["history"],
            "escalade_required": True,
            "escalade_type": "default"
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_all_memory")
async def clear_all_memory():
    try:
        global memory_store
        memory_store.clear()
        return {"status": "success", "message": "All memory cleared"}
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)