import logging
import uuid
import os
import datetime
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
from vector_db import QdrantStorage
from data_loader import embed_texts, \
    truncate_text  # Assure-toi que truncate_text n'est PAS importé de data_loader si tu le définis ici
from custom_types import RAGSearchResult
from parsing_utils import parse_ccq_pdf

# On charge les variables d'environnement
load_dotenv()

# Configuration du client Inngest
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    # Permet de basculer en mode prod si besoin via variable d'env
    is_production=os.getenv("INNGEST_IS_PROD", "False") == "True",
    event_key=os.getenv("INNGEST_EVENT_KEY", "local_dev_key"),
)

# --- FONCTION 1 : INGESTION ROBUSTE (VERSION BATCHING POUR GROS FICHIERS) ---
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF Heavy",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=1, period=datetime.timedelta(minutes=1))
)
async def rag_ingest_pdf(ctx: inngest.Context):
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)

    async def _process_heavy_job():
        print(f"🚀 [Job] Démarrage Ingestion Lourd : {pdf_path}")

        # 1. Parsing
        print("⏳ Parsing LlamaParse en cours...")
        chunks = await parse_ccq_pdf(pdf_path)
        total_chunks = len(chunks)
        print(f"✅ Parsing terminé : {total_chunks} chunks.")

        # 2. Batch Processing
        BATCH_SIZE = 100
        store = QdrantStorage()

        print(f"⚙️ Début du traitement par lots ({total_chunks} chunks)...")

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            current_batch = (i // BATCH_SIZE) + 1

            # --- DEBUG : Vérification des tailles avant coupe ---
            max_len_input = max([len(c["content"] or "") for c in batch])
            print(f"🔍 Batch {current_batch}: Taille max avant coupe = {max_len_input}")

            # A. Vectorisation (Embeddings) avec COUPE SÉCURISÉE
            # On utilise la fonction truncate_text définie plus haut
            texts = [truncate_text(c["content"]) for c in batch]

            # --- DEBUG : Vérification après coupe ---
            max_len_output = max([len(t) for t in texts])
            if max_len_output > 20000:
                print(f"🚨 ALERTE CRITIQUE : La coupe a échoué ! Max reste {max_len_output}")

            try:
                # Appel synchrone (pas de await car ta fonction embed_texts est standard)
                vecs = embed_texts(texts)
            except Exception as e:
                print(f"❌ Erreur critique vectorisation sur le batch {current_batch}")
                print(f"❌ Détail : {e}")
                raise e  # On arrête tout si ça plante ici

            # B. Préparation Qdrant
            ids = []
            payloads = []
            for j, item in enumerate(batch):
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i + j}"))
                ids.append(uid)
                # Attention : On stocke le texte COMPLET dans Qdrant (pour le RAG),
                # même si on a vectorisé une version coupée. C'est mieux pour la réponse finale.
                payloads.append({
                    "source": source_id,
                    "text": item["content"],
                    "metadata": item["metadata"]
                })

            # C. Sauvegarde
            store.upsert(ids, vecs, payloads)
            print(f"   ✅ Batch {current_batch} sauvegardé dans Qdrant.")

        return {"status": "success", "total_chunks": total_chunks}

    return await ctx.step.run("full_ingestion_process", _process_heavy_job)


# --- FONCTION 2 : RECHERCHE INTELLIGENTE (RAG) ---
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # 1. Recherche dans Qdrant
    def _search():
        q_vec = embed_texts([question])[0]
        res = QdrantStorage().search(q_vec, top_k)
        # On retourne un dictionnaire pur, facile à sérialiser pour Inngest
        return {
            "contexts": res["contexts"],
            "sources": res["sources"]
        }

    # On ne met PAS output_type=RAGSearchResult ici pour éviter les erreurs de validation
    found_dict = await ctx.step.run("search_vectors", lambda: _search())

    # On reconstruit l'objet APRES (si on en a besoin pour l'autocomplétion)
    # ou on utilise juste le dict directement.
    contexts = found_dict["contexts"]
    sources = found_dict["sources"]

    # 2. Préparation du contexte
    formatted_contexts = []
    # On utilise les variables extraites du dict
    for text, source_name in zip(contexts, sources):
        clean_source = str(source_name).replace("\\", "/").split("/")[-1]
        entry = f"DOCUMENT: {clean_source}\nEXTRAIT: {text}\n"
        formatted_contexts.append(entry)

    context_block = "\n---\n".join(formatted_contexts)

    # 3. Prompt Système Expert CCQ
    system_prompt = """
    Tu es un architecte expert en Code de Construction du Québec (CCQ/RBQ).

    MISSION :
    Ton rôle est d'analyser les règlements pour donner une réponse COMPLÈTE et NUANCÉE.

    RÈGLES D'ANALYSE (CRITIQUES) :
    1. Si la réponse provient d'un TABLEAU, ne donne pas juste un chiffre.
       -> Tu DOIS expliquer les CONDITIONS (ex: "Cela dépend de la distance limitative", "Selon la surface totale...").
       -> Si tu vois plusieurs colonnes, liste les scénarios possibles sous forme de bullet points.

    2. FORMAT DES CITATIONS :
       Pour chaque affirmation, ajoute la source précise :
       • [Fichier], Page [X]
         « [Citation courte] »

    3. Si le texte est ambigu ou incomplet, pose une question de clarification à l'utilisateur.

    EXEMPLE DE BONNE RÉPONSE :
    "Le pourcentage de baies vitrées permises dépend de la distance limitative avec la limite de propriété :
     - Si la distance est < 1,2m : 0% permis.
     - Si la distance est de 2,0m : 12% permis.
     [Source: CCQ-2010.pdf, Tableau 9.10.15.4, Page 634]"
    """

    user_content = (
        f"CONTEXTE FOURNI :\n{context_block}\n\n"
        f"QUESTION UTILISATEUR : {question}"
    )

    # 4. Appel à GPT-4o
    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()

    return {"answer": answer, "sources": sources, "num_contexts": len(contexts)}


# --- DÉMARRAGE APP ---
app = FastAPI()

# Enregistrement des fonctions Inngest
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])