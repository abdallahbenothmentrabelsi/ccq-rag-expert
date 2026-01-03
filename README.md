# 🏗️ CCQ RAG Expert - Assistant IA pour Architectes

![Status](https://img.shields.io/badge/Status-Beta-blue)
![Python](https://img.shields.io/badge/Python-3.11-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)

##  Description

**CCQ RAG Expert** est un assistant intelligent conçu pour les architectes et entrepreneurs au Québec. Il permet d'interroger en langage naturel les documents complexes du **Code de Construction du Québec (CCQ)** et de la **RBQ**.

Contrairement à un ChatGPT classique, cet outil utilise une architecture **RAG (Retrieval-Augmented Generation)** pour garantir :
*    **Zéro hallucination** : Il ne répond qu'avec les documents fournis.
*    **Citations précises** : Chaque réponse inclut le nom du fichier, le numéro de l'article et la page exacte.
*    **Données sécurisées** : Les documents sont traités localement ou sur serveur privé (Conforme Loi 25).

## ️ Stack Technique

Ce projet repose sur une architecture moderne et robuste ("Production Grade") :

*   **Backend API** : [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Orchestration** : [Inngest](https://www.inngest.com/) (Gestion des files d'attente et ingestion résiliente sans timeout)
*   **Base Vectorielle** : [Qdrant](https://qdrant.tech/) (Stockage des embeddings)
*   **LLM & Embeddings** : OpenAI (`gpt-4o-mini` & `text-embedding-3-small`)
*   **Frontend (Test)** : Streamlit
*   **Infrastructure** : Docker & Docker Compose

## 🚀 Installation & Démarrage

### Pré-requis
*   Docker & Docker Compose installés.
*   Python 3.10+ installé.
*   Une clé API OpenAI.

### 1. Cloner le projet
```bash
git clone https://github.com/abdallahbenothmentrabelsi/ccq-rag-expert.git
cd ccq-rag-expert
```

### 2. Configurer les variables d'environnement
Créez un fichier `.env` à la racine et ajoutez votre clé :
```ini
OPENAI_API_KEY=sk-votre-cle-api-ici
INNGEST_SIGNING_KEY= # Laisser vide en local
INNGEST_EVENT_KEY= # Laisser vide en local
```

### 3. Lancer l'infrastructure (Qdrant & Inngest)
```bash
docker-compose up -d
```
*   Accès Dashboard Qdrant : `http://localhost:6333/dashboard`
*   Accès Dashboard Inngest : `http://localhost:8288`

### 4. Lancer le Backend (API)
Il est conseillé d'utiliser un environnement virtuel.
```bash
# Création venv (Windows)
python -m venv venv
venv\Scripts\activate

# Installation dépendances
pip install -r requirements.txt

# Démarrage Serveur
uvicorn main:app --reload
```
*   Accès Swagger API : `http://localhost:8000/docs`

### 5. Lancer l'interface de Chat (Streamlit)
```bash
streamlit run streamlit_app.py
```

##  Utilisation

### Ingestion de documents (PDF)
Le système utilise Inngest pour traiter les gros fichiers PDF sans timeout.
1.  Déposez un fichier PDF dans le dossier `uploads/`.
2.  Envoyez un événement via le Dashboard Inngest :
    *   **Event Name:** `rag/ingest_pdf`
    *   **Data:** `{"pdf_path": "uploads/CCQ-2010.pdf", "source_id": "CCQ-2010"}`

### Posez une question
Utilisez l'interface Streamlit ou l'endpoint API `/chat`.

## ️ Architecture & Sécurité

*   **Chunking Intelligent** : Les documents sont découpés en gardant le contexte des numéros de pages.
*   **Batch Processing** : L'envoi vers OpenAI et Qdrant se fait par paquets (batchs) pour éviter les erreurs de limites API.
*   **Filtrage** : Le système est prêt pour le Multi-Tenant (séparation des données par utilisateur).

## Licence
Propriété de Nodeluck - Tous droits réservés.
