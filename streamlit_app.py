import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def run_async(coro):
    """
    Exécute une coroutine de manière sûre dans Streamlit.
    Crée une nouvelle boucle si nécessaire pour éviter les conflits.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# --- CONFIGURATION INNGEST ---

# ⚠️ IMPORTANT : Pas de @st.cache_resource ici !
# Cela évite de garder un client lié à une boucle d'événement fermée.
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
        # Le SDK va lire automatiquement INNGEST_BASE_URL dans les variables d'env
        event_key=os.getenv("INNGEST_EVENT_KEY", "local_dev_key")
    )


def _get_api_root_url() -> str:
    """Récupère l'URL racine d'Inngest (compatible Docker & Local)"""
    # 1. Priorité à la variable Docker standard
    url = os.getenv("INNGEST_BASE_URL")

    # 2. Compatibilité arrière
    if not url:
        url = os.getenv("INNGEST_API_BASE")

    # 3. Fallback Local (quand tu testes sur ton PC sans Docker)
    if not url:
        url = "http://127.0.0.1:8288"

    return url.rstrip("/")

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def fetch_runs(event_id: str) -> list[dict]:
    # Construction intelligente de l'URL
    base = _get_api_root_url()

    # Si l'URL de base ne contient pas /v1, on l'ajoute si nécessaire
    # (L'API Inngest standard est souvent sur /v1/events/...)
    if "/v1" not in base:
        url = f"{base}/v1/events/{event_id}/runs"
    else:
        url = f"{base}/events/{event_id}/runs"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        # On affiche l'erreur discrètement dans les logs ou via st.error si critique
        print(f"Erreur polling Inngest ({url}): {e}")
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None

    # Barre de progression pour le feedback utilisateur
    progress_bar = st.progress(0, text="En attente du démarrage...")

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status == "Running":
                progress_bar.progress(50, text="L'IA réfléchit...")

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                progress_bar.progress(100, text="Terminé !")
                time.sleep(0.5)  # Petit délai pour voir le 100%
                progress_bar.empty()
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                raise RuntimeError(f"Le traitement a échoué (Statut: {status})")

        if time.time() - start > timeout_s:
            progress_bar.empty()
            raise TimeoutError(f"Trop long... (Dernier statut: {last_status})")

        time.sleep(poll_interval_s)

st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    # Bouton explicite pour éviter les upload auto intempestifs
    if st.button("Lancer l'ingestion"):
        with st.spinner("Envoi du fichier..."):
            path = save_uploaded_pdf(uploaded)
            # Utilisation de notre wrapper safe
            run_async(send_rag_ingest_event(path))
            time.sleep(0.5)
        st.success(f"Ingestion déclenchée pour : {path.name}")
        st.caption("Regarde le Dashboard Inngest pour voir la progression.")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        try:
            with st.spinner("Envoi de la question..."):
                # 1. Envoi asynchrone sécurisé
                event_id = run_async(send_rag_query_event(question.strip(), int(top_k)))

            # 2. Attente active (Polling synchrone)
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

            st.subheader("Answer")
            st.write(answer or "(Pas de réponse générée)")

            if sources:
                with st.expander("Voir les sources"):
                    for s in sources:
                        st.write(f"- {s}")

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
