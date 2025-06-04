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

# Base de données complète extraite du JSON
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
        "bloc_A": {
            "id": "suivi_entreprise",
            "response": "Bien noté 👌\nPour te répondre au mieux, est-ce que tu sais :\n• Environ quand la formation s'est terminée ?\n• Et comment elle a été financée (par une entreprise directement, un OPCO, ou autre) ?\n👉 Une fois que j'ai ça, je te dis si le délai est normal ou si on doit faire une vérification ensemble 😊"
        },
        "bloc_B": {
            "id": "ambassadeur_nouveau",
            "response": "Salut 😄\nOui, on propose un programme super simple pour gagner de l'argent en nous envoyant des contacts intéressés 💼 💸\n\n🎯 Voici comment ça marche :\n✅ 1. Tu t'abonnes à nos réseaux pour suivre les actus\n👉 Insta : https://hi.switchy.io/InstagramWeiWei\n👉 Snap : https://hi.switchy.io/SnapChatWeiWei\n\n✅ 2. Tu nous transmets une liste de contacts\nNom + prénom + téléphone ou email (SIRET si c'est pro, c'est encore mieux)\n🔗 Formulaire : https://mrqz.to/AffiliationPromotion\n\n✅ 3. Tu touches une commission jusqu'à 60 % si un dossier est validé 🤑\nTu peux être payé directement sur ton compte perso (jusqu'à 3000 €/an et 3 virements)\nAu-delà, on peut t'aider à créer une micro-entreprise, c'est rapide ✨\n\nTu veux qu'on t'aide à démarrer ? Ou tu veux déjà envoyer ta première liste ? 📲"
        },
        "bloc_C": {
            "id": "cpf_indisponible",
            "response": "Salut 👋\nPour le moment, nous ne faisons plus de formations financées par le CPF 🚫\n👉 Par contre, on continue d'accompagner les professionnels, entreprises, auto-entrepreneurs ou salariés grâce à d'autres dispositifs de financement 💼\n\nSi tu veux être tenu au courant dès que les formations CPF reviennent, tu peux t'abonner ici 👇\n🔗 Insta : https://hi.switchy.io/InstagramWeiWei\n🔗 Snap : https://hi.switchy.io/SnapChatWeiWei\n\nTu veux qu'on t'explique comment ça fonctionne pour les pros ? Ou tu connais quelqu'un à qui ça pourrait servir ? 😊"
        },
        "bloc_D": {
            "id": "ambassadeur_demande",
            "response": "Salut 😄\nTu veux devenir ambassadeur et commencer à gagner de l'argent avec nous ? C'est super simple 👇\n\n✅ Étape 1 : Tu t'abonnes à nos réseaux\n👉 Insta : https://hi.switchy.io/InstagramWeiWei\n👉 Snap : https://hi.switchy.io/SnapChatWeiWei\n\n✅ Étape 2 : Tu nous envoies une liste de contacts intéressés (nom, prénom, téléphone ou email).\n➕ Si c'est une entreprise ou un pro, le SIRET est un petit bonus 😉\n🔗 Formulaire ici : https://mrqz.to/AffiliationPromotion\n\n✅ Étape 3 : Si un dossier est validé, tu touches une commission jusqu'à 60 % 🤑\nEt tu peux même être payé sur ton compte perso (jusqu'à 3000 €/an et 3 virements)\n\nTu veux qu'on t'aide à démarrer ou tu envoies ta première liste ? 📲"
        },
        "bloc_E": {
            "id": "transmission_contacts",
            "response": "Super 😄\nPour nous envoyer des contacts, c'est très simple 👇\n\n✅ Il nous faut le nom, le prénom, et un contact (téléphone ou email)\n➕ Si c'est une entreprise ou un pro, le SIRET est un petit bonus qui nous aide beaucoup 😉\n\n🔗 Tu peux les transmettre directement ici :\nhttps://mrqz.to/AffiliationPromotion\n\nTu peux nous en envoyer un seul ou une liste complète, comme tu préfères ✨\nEt si tu as le moindre souci ou une question, je suis là pour t'aider 👍"
        },
        "bloc_F": {
            "id": "paiement_formation",
            "response": "Salut 👋\nLe délai dépend du type de formation qui va être rémunérée et surtout de la manière dont elle a été financée 💡\n\n🔹 Si la formation a été payée directement (par un particulier ou une entreprise)\n→ Le paiement est effectué sous 7 jours après la fin de la formation et réception du dossier complet (émargement + administratif) 🧾\n\n🔹 Si la formation a été financée par le CPF\n→ Le paiement se fait à partir de 45 jours, mais uniquement à compter de la réception effective des feuilles d'émargement signées ✍\n\n🔹 Si le dossier est financé via un OPCO\n→ Le délai moyen est de 2 mois, mais certains dossiers prennent jusqu'à 6 mois ⏳\n\nPour que je puisse t'aider au mieux, est-ce que tu peux me préciser :\n• Comment la formation a été financée ?\n• Et environ quand elle s'est terminée ?\n\n👉 Je te dirai si c'est encore dans les délais normaux, ou si on fait une vérification 😊"
        },
        "bloc_G": {
            "id": "demande_humain",
            "response": "Bien sûr 👋\nJe suis là pour t'aider au maximum 😊\n\nTu peux me dire rapidement ce que tu veux savoir ou faire ?\n👉 Si je peux répondre directement, je le fais tout de suite ✅\nSinon, je ferai suivre à la bonne personne dans l'équipe.\n\n🕐 Juste pour info : notre équipe est dispo du lundi au vendredi, de 9h à 17h (hors pause déjeuner)\nDonc selon le moment, il peut y avoir un petit délai de traitement.\n\nMerci d'avance pour ta patience 🙏"
        },
        "bloc_H": {
            "id": "script_prospect",
            "response": "Tu peux lui dire quelque chose comme :\n\n\"Je travaille avec un organisme de formation super sérieux. Ils proposent des formations personnalisées, souvent 100 % financées, et ils s'occupent de tout, vraiment.\n\nL'entreprise n'a rien à avancer, ni à gérer. Et moi je peux être rémunéré si ça se met en place.\n\nSi ça t'intéresse, je te mets en contact avec eux pour voir ce qui est possible 💬 \""
        },
        "bloc_I1": {
            "id": "argumentaire_entreprise",
            "response": "\"Je vous parle d'un organisme de formation qui s'occupe de tout :\n\n✅ Étude du budget\n✅ Proposition de formations sur mesure\n✅ Gestion 100 % prise en charge (OPCO, FAF, etc.)\n\nVous n'avez rien à avancer 💸\n\nEn plus, ça vous permet de :\n• Répondre à votre obligation légale de former vos salariés (Code du travail – art. L6321-1)\n• Faire monter vos équipes en compétences\n• Optimiser votre budget formation (il est renouvelé chaque année et souvent perdu s'il n'est pas utilisé)\n\n👉 Si ça vous intéresse, je vous mets en relation avec une équipe qui va tout gérer pour vous, gratuitement et rapidement.\""
        },
        "bloc_I2": {
            "id": "argumentaire_ambassadeur",
            "response": "\"C'est une opportunité hyper simple pour gagner de l'argent tous les ans.\n\nTu recommandes une entreprise ➡ On s'en occupe\n✅ L'entreprise n'a rien à payer\n✅ Elle est obligée de former ses salariés chaque année (c'est une loi)\n✅ On lui trouve une formation utile (bureautique, vente, langues, etc.)\n✅ On gère tout : paperasse, appel, financement\n\nEt toi, si le dossier passe, tu touches une commission jusqu'à 60 % 🤑\n\nEt le meilleur dans tout ça ?\n➕ Le budget formation est renouvelé chaque année\n➕ Donc tu peux être payé chaque année avec le même client !\n\nTu veux que je t'aide à identifier une entreprise autour de toi ?\""
        },
        "bloc_J": {
            "id": "delai_global",
            "response": "Bonne question 👇\n\nEn moyenne, il faut compter entre 3 et 6 mois ⏳\n\nÇa dépend surtout :\n✅ du type de financement (CPF, OPCO, entreprise directe)\n✅ de la réactivité du contact (émargements, validation, etc.)\n✅ du temps de traitement de l'organisme de financement\n\n📌 Exemple :\nSi c'est une entreprise qui paie directement → paiement en 7 jours après la formation\nSi c'est un dossier CPF → il faut souvent attendre 45 jours mini après les signatures\nPour les OPCO → c'est en moyenne 2 mois, parfois plus\n\n🧠 C'est pour ça qu'on te conseille d'envoyer plusieurs contacts au début → pour que les paiements s'enchaînent ensuite 🔁\nEt nous, on gère tout le suivi administratif entre temps 👌"
        }
    },
    "blocs_escalade": {
        "escalade_agent_admin": {
            "id": "escalade_agent_admin",
            "response": "🔁 ESCALADE AGENT ADMIN\n\n📅 Rappel :\n\"Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nNous te répondrons dès que possible.\"\n\n🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nOn te tiendra informé dès qu'on a du nouveau ✅"
        },
        "escalade_agent_co": {
            "id": "escalade_agent_co",
            "response": "📞 ESCALADE AGENT CO\n\n📅 Rappel :\n\"Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nNous te répondrons dès que possible.\"\n\n🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nOn te tiendra informé dès que possible ✅"
        }
    },
    "blocs_complementaires": {
        "relance_apres_escalade": {
            "id": "relance_apres_escalade",
            "response": "Je comprends que c'est long 😔 Notre équipe fait le max, mais comme c'est un traitement humain, ça dépend un peu de l'heure et du jour.\n\nIls bossent du lundi au vendredi, entre 9h et 17h (hors pause déjeuner).\n\nJe leur fais un petit rappel pour qu'ils reprennent contact avec toi dès que possible 🙏"
        },
        "sans_reseaux_sociaux": {
            "id": "sans_reseaux_sociaux",
            "response": "Pas de souci si tu n'es pas sur Insta ou Snap 😌 Tu peux simplement nous envoyer des contacts potentiellement intéressés. Ça fonctionne très bien aussi 😉"
        },
        "legalite_programme": {
            "id": "legalite_programme",
            "response": "On ne peut pas inscrire une personne dans une formation si son but est d'être rémunérée pour ça. En revanche, si tu fais la formation sérieusement, tu peux ensuite participer au programme d'affiliation et parrainer d'autres personnes."
        },
        "micro_entreprise": {
            "id": "micro_entreprise",
            "response": "Au-delà de 3000 € ou 3 virements, il faudra créer une micro-entreprise. Mais t'inquiète, c'est super simple, et on pourra t'accompagner étape par étape ✨"
        },
        "transmission_contacts_requis": {
            "id": "transmission_contacts_requis",
            "response": "Tu peux nous envoyer le nom, prénom, et un contact (téléphone ou email). Si tu as aussi leur SIRET, c'est top 😉 (ça nous aide pour les pros). Pas grave sinon, on fera sans."
        },
        # Sous-blocs du bloc F (paiement_formation)
        "cpf_bloque": {
            "id": "cpf_bloque",
            "response": "Ce dossier fait partie des quelques cas bloqués depuis la réforme CPF de février 2025.\n\n✅ Tous les éléments nécessaires ont bien été transmis à l'organisme de contrôle 📄 🔍\n❌ Mais la Caisse des Dépôts met souvent plusieurs semaines (parfois jusqu'à 2 mois) pour redemander un document après en avoir reçu un autre.\n\n👉 On accompagne au maximum le centre de formation pour que tout rentre dans l'ordre.\n🙏 On est aussi impactés financièrement, car chaque formation a un coût pour nous.\n\n💪 On garde confiance et on espère une issue favorable très bientôt.\n🗣 Et on s'engage à revenir vers toi dès qu'on a du nouveau. Merci pour ta patience 🙏"
        },
        "filtrage_cpf_bloque": {
            "id": "filtrage_cpf_bloque",
            "response": "Juste avant que je transmette ta demande 🙏\n\nEst-ce que tu as déjà été informé par l'équipe que ton dossier CPF faisait partie des quelques cas bloqués par la Caisse des Dépôts ?\n\n👉 Si oui, je te donne directement toutes les infos liées à ce blocage.\nSinon, je fais remonter ta demande à notre équipe pour vérification ✅"
        },
        "opco_delai_depasse": {
            "id": "opco_delai_depasse",
            "response": "Merci pour ta réponse 🙏\n\nPour un financement via un OPCO, le délai moyen est de 2 mois. Certains dossiers peuvent aller jusqu'à 6 mois ⏳\n\nMais vu que cela fait plus de 2 mois, on préfère ne pas te faire attendre plus longtemps sans retour.\n\n👉 Je vais transmettre ta demande à notre équipe pour qu'on vérifie ton dossier dès maintenant 🧾\n\n🔁 ESCALADE AGENT ADMIN\n🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nOn te tiendra informé dès qu'on a une réponse ✅"
        },
        "anomalie_probable": {
            "id": "anomalie_probable",
            "response": "Merci pour ta réponse 🙏\n\nSi les délais habituels sont dépassés et que ton dossier ne fait pas partie des cas bloqués, je vais le faire remonter à notre équipe pour qu'on vérifie tout de suite ce qui se passe 🧾\n\n🔁 ESCALADE AGENT ADMIN\n🕐 Notre équipe traite les demandes du lundi au vendredi, de 9h à 17h (hors pause déjeuner).\nOn te tiendra informé dès que possible ✅"
        }
    }
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

# Initialisation du vector store (lazy-loaded dans la fonction)
def initialize_vector_store():
    texts = [bloc["response"] for bloc in blocs]
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)
    return FAISS.from_documents(docs, embeddings)

vector_store = initialize_vector_store()

# Fonction pour détecter les problèmes de paiement (priorité absolue)
def detect_payment_issue(message):
    payment_keywords = [
        "j'ai pas été payé", "toujours rien reçu", "je devais avoir un virement",
        "on m'a dit que j'allais être payé", "ça fait 3 mois j'attends",
        "une attente d'argent", "plainte sur non-versement", "retard de paiement",
        "promesse non tenue", "pas encore payé", "virement en retard",
        "je vais être payé quand", "ça fait 20 jours j'aurais dû être payé",
        "toujours pas reçu virement", "retard paiement formation"
    ]
    
    message_lower = message.lower()
    for keyword in payment_keywords:
        if keyword in message_lower:
            return True
    return False

# Fonction pour analyser le contexte et choisir le bon sous-bloc
def get_contextualized_response(user_message, matched_bloc):
    if matched_bloc["id"] == "paiement_formation":
        # Logique pour les sous-blocs du paiement
        message_lower = user_message.lower()
        
        # Vérification des mots-clés pour différents cas
        if any(keyword in message_lower for keyword in ["cpf", "compte personnel"]):
            return {
                "matched_bloc_response": blocs_data["blocs_complementaires"]["filtrage_cpf_bloque"]["response"],
                "escalade_required": False
            }
        elif any(keyword in message_lower for keyword in ["opco", "entreprise"]):
            return {
                "matched_bloc_response": matched_bloc["response"],
                "escalade_required": True,
                "escalade_type": "admin"
            }
    
    return {
        "matched_bloc_response": matched_bloc["response"],
        "escalade_required": False
    }

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
        
        # Détection prioritaire des problèmes de paiement
        if detect_payment_issue(user_message):
            payment_bloc = next((bloc for bloc in blocs if bloc["id"] == "paiement_formation"), None)
            if payment_bloc:
                contextualized = get_contextualized_response(user_message, payment_bloc)
                memory.chat_memory.add_ai_message(contextualized["matched_bloc_response"])
                
                response = {
                    "matched_bloc_response": contextualized["matched_bloc_response"],
                    "memory": memory.load_memory_variables({})["history"],
                    "priority_detection": "PAYMENT_ISSUE"
                }
                
                if contextualized.get("escalade_required"):
                    response["escalade_required"] = True
                    response["escalade_type"] = contextualized.get("escalade_type", "admin")
                
                return response
        
        # Recherche sémantique pour trouver un bloc pertinent
        similar_docs = vector_store.similarity_search(user_message, k=3)
        best_match = None
        best_score = 0
        
        for doc in similar_docs:
            for bloc in blocs:
                if bloc["response"] == doc.page_content:
                    # Scoring simple basé sur la longueur du texte similaire
                    score = len(doc.page_content)
                    if score > best_score:
                        best_score = score
                        best_match = bloc
                    break
        
        if best_match:
            contextualized = get_contextualized_response(user_message, best_match)
            memory.chat_memory.add_ai_message(contextualized["matched_bloc_response"])
            
            response = {
                "matched_bloc_response": contextualized["matched_bloc_response"],
                "memory": memory.load_memory_variables({})["history"],
                "bloc_id": best_match["id"],
                "bloc_category": best_match["category"]
            }
            
            if contextualized.get("escalade_required"):
                response["escalade_required"] = True
                response["escalade_type"] = contextualized.get("escalade_type", "admin")
            
            return response
        
        # Si rien n'est trouvé, escalade par défaut
        escalade_response = "Je vais faire suivre à la bonne personne dans l'équipe 😊 Notre équipe est disponible du lundi au vendredi, de 9h à 17h (hors pause déjeuner)."
        memory.chat_memory.add_ai_message(escalade_response)
        
        return {
            "matched_bloc_response": escalade_response,
            "memory": memory.load_memory_variables({})["history"],
            "escalade_required": True,
            "escalade_type": "default"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour obtenir la liste des blocs disponibles
@app.get("/blocs")
async def get_blocs():
    return {
        "total_blocs": len(blocs),
        "blocs": [{"id": bloc["id"], "category": bloc["category"]} for bloc in blocs],
        "categories": list(set(bloc["category"] for bloc in blocs))
    }

# Endpoint pour tester un message spécifique
@app.post("/test")
async def test_message(request: Request):
    try:
        body = await request.json()
        test_message = body.get("message", "")
        
        # Test de détection de paiement
        payment_detected = detect_payment_issue(test_message)
        
        # Recherche sémantique
        similar_docs = vector_store.similarity_search(test_message, k=3)
        matches = []
        
        for doc in similar_docs:
            for bloc in blocs:
                if bloc["response"] == doc.page_content:
                    matches.append({
                        "bloc_id": bloc["id"],
                        "category": bloc["category"],
                        "response_preview": bloc["response"][:100] + "..."
                    })
                    break
        
        return {
            "test_message": test_message,
            "payment_issue_detected": payment_detected,
            "semantic_matches": matches[:3],
            "total_matches": len(matches)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)