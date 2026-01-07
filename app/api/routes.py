from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.rag_service import generate_rag_response  # La fonction qu'on a créée avant
# from app.schemas.custom_types import ChatRequest  # Si tu as défini des classes

# On crée un "mini-app" qu'on appelle router
router = APIRouter()

# --- C'est ici que tu colles tes anciennes routes ---

@router.post("/chat")  # Remplace @app par @router
async def chat_endpoint(question: str):
    try:
        # Appelle la logique intelligente (qui est maintenant dans services/rag_service.py)
        reponse = await generate_rag_response(question)
        return {"reponse": reponse}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")  # Remplace @app par @router
async def ingest_endpoint(file: UploadFile = File(...)):
    # Mets ici ton code d'upload qui était dans l'ancien main.py
    return {"message": f"Fichier {file.filename} reçu"}
