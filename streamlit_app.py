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


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """
    Configure le client Inngest pour qu'il fonctionne dans Docker
    en pointant vers le conteneur inngest-dev.
    """
    event_key = os.getenv("INNGEST_EVENT_KEY", "local_dev_key")

    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
        event_key=event_key
    )


def _get_api_root_url() -> str:
    """
    Récupère l'URL racine pour nos appels manuels (Polling).
    Doit correspondre à ce que le SDK utilise.
    """
    # 1. On priorise la nouvelle variable standard du SDK
    url = os.getenv("INNGEST_BASE_URL")

    # 2. Si elle n'existe pas, on cherche l'ancienne (compatibilité)
    if not url:
        url = os.getenv("INNGEST_API_BASE")

    # 3. Fallback localhost pour tes tests hors Docker
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
    # Construction manuelle de l'URL pour l'API REST du Dev Server
    # Ici on doit souvent ajouter /v1 si l'URL de base ne l'a pas
    base = _get_api_root_url()

    # Si l'URL de base contient déjà /v1, on ne l'ajoute pas
    if "/v1" in base:
        url = f"{base}/events/{event_id}/runs"
    else:
        url = f"{base}/v1/events/{event_id}/runs"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        st.error(f"Erreur de connexion à Inngest ({url}): {e}")
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 300.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None

    progress_bar = st.progress(0, text="En attente du démarrage...")

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            # Mise à jour visuelle
            if status == "Running":
                progress_bar.progress(50, text="Traitement en cours...")

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                progress_bar.progress(100, text="Terminé !")
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                raise RuntimeError(f"Le run a échoué avec le statut : {status}")

        if time.time() - start > timeout_s:
            progress_bar.empty()
            raise TimeoutError(f"Délai dépassé (Dernier statut: {last_status})")

        time.sleep(poll_interval_s)

st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    if st.button("Lancer l'ingestion"):
        with st.spinner("Envoi du fichier et déclenchement..."):
            path = save_uploaded_pdf(uploaded)
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(0.5)
        st.success(f"Ingestion déclenchée pour : {path.name}")
        st.info("Regarde le Dashboard Inngest pour voir la progression.")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Envoi de la question..."):
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))

        try:
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
            st.error(f"Erreur lors de la récupération de la réponse : {e}")
