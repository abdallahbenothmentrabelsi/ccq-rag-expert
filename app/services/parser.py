import os
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document

# Permet d'exécuter des boucles asynchrones imbriquées (nécessaire avec FastAPI/Inngest)
nest_asyncio.apply()


async def parse_ccq_pdf(file_path: str):
    """
    Transforme un PDF complexe (CCQ) en chunks Markdown structurés.

    Args:
        file_path (str): Chemin vers le fichier PDF temporaire.

    Returns:
        list[dict]: Liste de chunks prêts pour l'embedding.
    """

    # 1. Configuration du Parser LlamaParse
    # On force le format markdown et la langue française pour optimiser l'OCR
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # Récupère la clé du .env
        result_type="markdown",  # Essentiel pour les tableaux
        language="fr",  # Améliore la reconnaissance du français
        verbose=True
    )

    print(f"⏳ [LlamaParse] Début du parsing pour : {file_path}")

    # 2. Extraction du contenu (Appel API vers LlamaCloud)
    # LlamaParse retourne une liste de documents (pages/fichiers)
    documents = await parser.aload_data(file_path)

    # On combine tout le texte en un seul bloc Markdown
    full_markdown_text = "\n\n".join([doc.text for doc in documents])

    # Création d'un Document LlamaIndex pour le traitement suivant
    base_doc = Document(text=full_markdown_text, metadata={"source": file_path})

    # 3. Chunking Sémantique (MarkdownNodeParser)
    # Au lieu de couper arbitrairement, on coupe par Headers (# Titre, ## Sous-titre)
    node_parser = MarkdownNodeParser()

    # Cette fonction retourne une liste de "Nodes" (chunks intelligents)
    nodes = node_parser.get_nodes_from_documents([base_doc])

    # 4. Formatage pour ta base de données (Qdrant)
    # On transforme les objets complexes LlamaIndex en dictionnaires simples
    clean_chunks = []

    for node in nodes:
        # On récupère le texte du chunk
        content = node.get_content()

        # On récupère les métadonnées précieuses (ex: quel est le titre parent ?)
        metadata = node.metadata.copy()

        # RÉCUPÉRATION ROBUSTE DE LA PAGE
        # LlamaParse met parfois la page dans 'page_label' (string "42") ou 'page_number' (int 42)
        page = metadata.get("page_label") or metadata.get("page_number") or "N/A"

        # INJECTION DANS LE TEXTE (C'est ça le secret !)
        # On force le texte à commencer par "PAGE: 42"
        enriched_content = f"PAGE: {page}\nSOURCE: {file_path}\nCONTENU:\n{content}"

        # Ajout explicite pour le débogage ou le filtrage futur
        chunk_data = {
            "content": enriched_content,
            "metadata": metadata,
            "chunk_size": len(content)
        }
        clean_chunks.append(chunk_data)

    print(f"✅ [LlamaParse] Terminé : {len(clean_chunks)} chunks sémantiques générés.")
    return clean_chunks