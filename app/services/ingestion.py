import uuid
import datetime
import inngest
from app.core.inngest_client import inngest_client
from app.db.vector_store import QdrantStorage
from app.services.parser import parse_ccq_pdf
from app.services.text_utils import embed_texts, truncate_text


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF Heavy",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=1, period=datetime.timedelta(minutes=1))
)
async def rag_ingest_pdf(ctx: inngest.Context):
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)

    async def _process_heavy_job():
        print(f"🚀 [Job] Démarrage Ingestion : {pdf_path}")
        chunks = await parse_ccq_pdf(pdf_path)
        total_chunks = len(chunks)

        BATCH_SIZE = 100
        store = QdrantStorage()

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            texts = [truncate_text(c["content"]) for c in batch]
            vecs = embed_texts(texts)

            ids = []
            payloads = []
            for j, item in enumerate(batch):
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i + j}"))
                ids.append(uid)
                payloads.append({
                    "source": source_id,
                    "text": item["content"],
                    "metadata": item["metadata"]
                })

            store.upsert(ids, vecs, payloads)

        return {"status": "success", "total_chunks": total_chunks}

    return await ctx.step.run("full_ingestion_process", _process_heavy_job)
