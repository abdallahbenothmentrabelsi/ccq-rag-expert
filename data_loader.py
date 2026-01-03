from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# On initialise le client avec la clé API explicite (plus sûr)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modèle performant (mais attention, le 'large' coûte plus cher et est plus lourd)
# Si tu veux aller plus vite/moins cher, utilise "text-embedding-3-small" (dim 1536)
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

# Découpage du texte : des morceaux de ~1000 caractères avec chevauchement
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    """Charge le PDF en gardant la trace des PAGES."""
    print(f"📖 Lecture du PDF: {path}")

    # 1. On charge les documents (chaque 'doc' est souvent une page)
    docs = PDFReader().load_data(file=path)

    final_chunks = []

    # 2. On parcourt chaque page chargée
    for page_doc in docs:
        # LlamaIndex stocke souvent le num de page dans metadata
        # Sinon, on essaie de le deviner ou on met "Inconnu"
        page_num = page_doc.metadata.get("page_label") or page_doc.metadata.get("page_number") or "Inconnue"

        text = page_doc.text
        if not text:
            continue

        # 3. On découpe cette page en morceaux
        page_chunks = splitter.split_text(text)

        # 4. ASTUCE CRUCIALE : On colle l'info "Page X" DANS le texte du chunk
        # Comme ça, l'info est inséparable du texte, et GPT la verra toujours.
        for chunk in page_chunks:
            enriched_chunk = f"PAGE: {page_num}\nCONTENU: {chunk}"
            final_chunks.append(enriched_chunk)

    print(f"✂️  PDF découpé en {len(final_chunks)} morceaux enrichis avec numéros de page.")
    return final_chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Vectorise une liste de textes en envoyant des requêtes par paquets (batchs)
    pour ne pas dépasser la limite de tokens d'OpenAI.
    """
    all_embeddings = []
    # Taille du paquet : 50 morceaux par envoi. C'est un bon compromis sécurité/vitesse.
    batch_size = 50

    print(f"🚀 Démarrage vectorisation ({len(texts)} chunks) avec modèle {EMBED_MODEL}...")

    # On boucle de 0 à la fin, par pas de 50
    for i in range(0, len(texts), batch_size):
        # On extrait le lot actuel (ex: morceaux 0 à 50)
        batch = texts[i: i + batch_size]

        try:
            # On envoie ce petit lot à OpenAI
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
            # On extrait les vecteurs reçus
            batch_vecs = [item.embedding for item in response.data]

            # On les ajoute à la liste totale
            all_embeddings.extend(batch_vecs)

            print(f"   ✅ Batch {i // batch_size + 1} traité ({len(batch)} textes)")

        except Exception as e:
            print(f"❌ ERREUR sur le batch {i}: {str(e)}")
            # En prod, on pourrait réessayer, mais ici on lève l'erreur pour voir le problème
            raise e

    print(f"✨ Vectorisation terminée : {len(all_embeddings)} vecteurs générés.")
    return all_embeddings
