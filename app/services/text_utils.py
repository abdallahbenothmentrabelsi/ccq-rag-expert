from openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from app.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-large"
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 50
    print(f"🚀 Démarrage vectorisation ({len(texts)} chunks)...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
            print(f"   ✅ Batch {i // batch_size + 1} traité.")
        except Exception as e:
            print(f"❌ ERREUR sur le batch {i}: {str(e)}")
            raise e
    return all_embeddings


def truncate_text(text: str, max_chars: int = 12000) -> str:
    if text is None: return ""
    if len(text) <= max_chars: return text
    print(f"🔪 COUPURE : {len(text)} -> {max_chars} chars.")
    return text[:max_chars]
