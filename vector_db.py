from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        from qdrant_client.models import PointStruct

        # Batch size de sécurité pour Qdrant
        BATCH_SIZE = 100

        total = len(ids)
        print(f"💾 QDRANT: Insertion de {total} points en paquets de {BATCH_SIZE}...")

        for i in range(0, total, BATCH_SIZE):
            # On découpe en tranches
            batch_ids = ids[i: i + BATCH_SIZE]
            batch_vectors = vectors[i: i + BATCH_SIZE]
            batch_payloads = payloads[i: i + BATCH_SIZE]

            points = [
                PointStruct(id=idx, vector=vector, payload=payload)
                for idx, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]

            try:
                self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                    wait=True  # Important pour être sûr que c'est écrit
                )
                print(f"   ✅ Qdrant: Paquet {i // BATCH_SIZE + 1} inséré.")
            except Exception as e:
                print(f"❌ Qdrant Error batch {i}: {e}")
                raise e

    def search(self, query_vector, top_k: int = 5):
        # CORRECTION : Utilisation de query_points au lieu de search
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,  # Attention: paramètre renommé en 'query'
            with_payload=True,
            limit=top_k
        ).points  # On récupère la liste .points

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}