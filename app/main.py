import inngest.fast_api
from app.core.config import settings
from app.core.inngest_client import inngest_client
from app.services.ingestion import rag_ingest_pdf
from app.services.rag_service import rag_query_pdf_ai
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI(title=settings.PROJECT_NAME)

# On enregistre les fonctions Inngest importées depuis leurs modules respectifs
inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai]
)
# On rend le dossier "uploads" accessible publiquement via l'URL /static
# Attention : En prod, assure-toi que tu ne mets pas de fichiers confidentiels ici !
app.mount("/static", StaticFiles(directory="uploads"), name="static")
@app.get("/")
def health_check():
    return {"status": "Inngest Worker Running 🚀"}
