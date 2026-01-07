import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

# Charge les variables (surtout pour le dev local)
load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


# --- UTILITAIRES ---

def run_async(coro):
    """Exécute une coroutine asyncio proprement dans Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_inngest_client() -> inngest.Inngest:
    """Crée un client léger juste pour envoyer des événements."""
    return inngest.Inngest(
        app_id="rag_app",
        is_production=os.getenv("INNGEST_IS_PROD", "False") == "True",
        event_key=os.getenv("INNGEST_EVENT_KEY", "local_dev_key")
    )


def _get_api_root_url() -> str:
    """Trouve l'URL du serveur Inngest Dev (Docker ou Local)."""
    # Priorité 1: Variable explicite (définie dans docker-compose)
    url = os.getenv("INNGEST_BASE_URL")

    # Priorité 2: Fallback local par défaut
    if not url:
        url = "http://127.0.0.1:8288"

    return url.rstrip("/")

def save_uploaded_pdf(file) -> Path:
    """Sauvegarde temporaire du fichier uploadé."""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


# --- ENVOI D'ÉVÉNEMENTS (DÉCLENCHEURS) ---

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
    # On envoie l'événement et on récupère les IDs générés
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    # result est une liste d'IDs d'événements, on prend le premier
    return result[0]


# --- POLLING (ATTENTE DE RÉPONSE) ---

def fetch_runs(event_id: str) -> list[dict]:
    """Interroge l'API Inngest pour savoir où en est le traitement."""
    base = _get_api_root_url()

    # L'API Inngest Dev a parfois /v1, parfois non selon les versions
    url = f"{base}/v1/events/{event_id}/runs"

    try:
        resp = requests.get(url, timeout=5)
        # Si 404, on essaie sans /v1 (compatibilité)
        if resp.status_code == 404:
            url = f"{base}/events/{event_id}/runs"
            resp = requests.get(url, timeout=5)

        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        # En dev, c'est souvent parce que le serveur Inngest n'est pas prêt
        print(f"Polling warning ({url}): {e}")
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 120.0) -> dict:
    """Boucle d'attente active jusqu'à ce que l'IA réponde."""
    start = time.time()
    progress_bar = st.progress(0, text="Envoi au serveur...")

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")

            if status == "Running":
                progress_bar.progress(50, text="🧠 L'IA analyse le Code de Construction...")

            if status in ("Completed", "Succeeded"):
                progress_bar.progress(100, text="Réponse reçue !")
                time.sleep(0.5)
                progress_bar.empty()
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                raise RuntimeError(f"Le traitement a échoué (Statut: {status})")

        if time.time() - start > timeout_s:
            progress_bar.empty()
            raise TimeoutError("Le serveur met trop de temps à répondre.")

        time.sleep(1.0)  # On vérifie chaque seconde


# --- INTERFACE UTILISATEUR (UI) ---

st.title("🏗️ CCQ RAG Expert")
st.caption("Posez vos questions sur le Code de Construction du Québec")

# Onglet Upload (caché par défaut dans un expander pour cleaner l'UI)
with st.expander("📂 Ajouter un document PDF (Ingestion)"):
    uploaded = st.file_uploader("Choisir un fichier PDF", type=["pdf"])
    if uploaded and st.button("Lancer l'ingestion"):
        with st.spinner("Envoi du fichier..."):
            path = save_uploaded_pdf(uploaded)
            run_async(send_rag_ingest_event(path))
        st.success(f"Ingestion démarrée pour : {path.name}")
        st.info("Vérifiez le dashboard Inngest pour le suivi.")

st.divider()

# Zone de Chat
with st.form("rag_query_form"):
    question = st.text_input("Votre question :", placeholder="Ex: Quelle est la hauteur minimale d'un garde-corps ?")
    top_k = st.slider("Précision (Nb de chunks)", 1, 10, 5)
    submitted = st.form_submit_button("Poser la question")

    if submitted and question.strip():
        try:
            with st.spinner("Communication avec l'IA..."):
                event_id = run_async(send_rag_query_event(question.strip(), top_k))

            # On attend la réponse
            output = wait_for_run_output(event_id)

            answer = output.get("answer", "Pas de réponse.")
            sources = output.get("sources", [])

            st.markdown("### 🤖 Réponse :")
            st.markdown(answer)

            if sources:
                st.markdown("---")
                st.markdown("#### 📚 Sources utilisées :")
                for src in set(sources):  # set() pour dédoublonner
                    st.caption(f"📄 {src}")

        except Exception as e:
            st.error(f"Erreur : {e}")
