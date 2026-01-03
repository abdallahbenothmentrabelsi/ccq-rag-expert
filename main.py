import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult

# On charge les variables d'environnement (Clé API OpenAI, etc.)
load_dotenv()

# Configuration du client Inngest
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    event_key=os.getenv("INNGEST_EVENT_KEY", "local_dev_key"),
    serializer=inngest.PydanticSerializer()
)


# --- FONCTION 1 : INGESTION ROBUSTE (CORRIGÉE ✅) ---
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2,
        period=datetime.timedelta(minutes=1)
    ),
    # On garde le rate limit pour éviter d'ingérer le même fichier 10 fois
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """
    Ingère un PDF en une seule étape atomique.
    Évite l'erreur 'step output size limit' car on ne renvoie pas les chunks à Inngest.
    """

    # On définit une fonction interne qui fait tout le travail sale en local
    def _ingest_process():
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        # 1. Chargement & Découpage
        print(f"📄 [Ingest] Chargement du fichier : {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)
        print(f"✂️ [Ingest] {len(chunks)} morceaux créés.")

        # 2. Vectorisation (OpenAI)
        print("🧠 [Ingest] Génération des embeddings...")
        vecs = embed_texts(chunks)

        # 3. Sauvegarde dans Qdrant
        print("💾 [Ingest] Sauvegarde dans Qdrant...")
        # Génération d'IDs uniques basés sur le nom du fichier + index
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vecs, payloads)

        # 4. Retour ultra-léger (Succès)
        return {
            "status": "completed",
            "file": source_id,
            "chunks_count": len(chunks)
        }

    # On appelle la fonction via Inngest en UNE SEULE étape
    result = await ctx.step.run("full_ingestion_job", _ingest_process)

    return result


# --- FONCTION 2 : RECHERCHE INTELLIGENTE (RAG) ---
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    # Fonction interne de recherche vectorielle
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # 1. Recherche dans Qdrant
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # 2. Préparation du contexte pour ChatGPT
    formatted_contexts = []
    for text, source_name in zip(found.contexts, found.sources):
        # Nettoyage du nom de fichier (on garde juste ccq.pdf, pas C:/Users/...)
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

    3. Si le texte est ambigu ou incomplet, pose une question de clarification à l'utilisateur 
       (ex: "Quelle est la distance avec le bâtiment voisin ?").

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

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


# --- DÉMARRAGE APP ---
app = FastAPI()

# Enregistrement des fonctions Inngest
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])