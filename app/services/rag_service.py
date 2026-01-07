import os

import inngest
from inngest.experimental import ai
from app.core.inngest_client import inngest_client
from app.core.config import settings
from app.db.vector_store import QdrantStorage
from app.services.text_utils import embed_texts


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # --- 1. Récupération de l'URL publique pour les liens ---
    # En local Docker, c'est souvent localhost:8000. En prod, ce sera ton domaine.
    # On l'injecte dans le prompt pour que GPT puisse construire les liens.
    api_base_url = os.getenv("API_PUBLIC_URL", "http://localhost:8000")

    def _search():
        # Attention: assure-toi que embed_texts est bien importé
        q_vec = embed_texts([question])[0]
        res = QdrantStorage().search(q_vec, top_k)
        return {"contexts": res["contexts"], "sources": res["sources"]}

    found_dict = await ctx.step.run("search_vectors", lambda: _search())

    contexts = found_dict["contexts"]
    sources = found_dict["sources"]  # Liste de noms de fichiers (ex: "CCQ.pdf")

    # Préparation du contexte pour le LLM
    formatted_contexts = []
    for text, source_name in zip(contexts, sources):
        # On nettoie le nom du fichier pour éviter les chemins bizarres
        clean_filename = str(source_name).split("/")[-1]
        entry = f"FICHIER: {clean_filename}\nEXTRAIT:\n{text}\n"
        formatted_contexts.append(entry)

    context_block = "\n---\n".join(formatted_contexts)

    # --- 2. PROMPT SYSTÈME OPTIMISÉ POUR LES LIENS PDF ---

    system_prompt = f"""
    ROLE:
    Tu es l'Expert en Code de Construction du Québec (CCQ/RBQ). Ta mission est de fournir des analyses réglementaires précises, complètes et actionnables pour des architectes et entrepreneurs.

    CONTEXTE ET FORMAT DE LIEN:
    URL de base API: {api_base_url}
    Format de lien OBLIGATOIRE: [📄 Voir Source (Page X)]({api_base_url}/static/NOM_FICHIER#page=NUMERO)
    (Pour trouver le NUMERO, cherche 'DOCUMENT_PAGE_NUMBER: X' ou 'PAGE: X' dans le texte fourni).

    DIRECTIVES DE RÉPONSE:
    1.  **Analyse Complète :** Ne donne jamais un chiffre isolé. Explique toujours les conditions, exceptions et nuances (ex: distance limitative, superficie).
    2.  **Tableaux Complexes :** Si l'info vient d'un tableau, liste TOUS les scénarios possibles via des bullet points clairs.
    3.  **Preuves & Liens :** Chaque affirmation doit être sourcée. Finis chaque paragraphe par le lien cliquable PDF vers la page exacte.
    4.  **Clarté :** Utilise un langage professionnel mais accessible. Structure avec des titres Markdown (##) et des listes.
    5.  **Honnêteté :** Si le contexte ne contient pas la réponse, dis "Désolé, cette information n'est pas dans les documents fournis." Ne l'invente pas.

    EXEMPLE DE STRUCTURE ATTENDUE:
    ### Titre de la règle
    Explication détaillée de la règle...
    - Condition A : Résultat X
    - Condition B : Résultat Y

    > [📄 Voir Source (CCQ-2010.pdf - Page 634)]({api_base_url}/static/CCQ-2010.pdf#page=634)
    """

    user_content = f"CONTEXTE:\n{context_block}\n\nQUESTION: {question}"

    adapter = ai.openai.Adapter(
        auth_key=settings.OPENAI_API_KEY,
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0  # On veut de la précision, pas de la créativité
        }
    )

    return {
        "answer": res["choices"][0]["message"]["content"],
        "sources": sources
    }