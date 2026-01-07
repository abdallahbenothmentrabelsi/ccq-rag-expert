# Utilise une image Python légère
FROM python:3.11-slim

# Définit le dossier de travail
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y build-essential

# Copie les fichiers de dépendances
COPY requirements.txt .

# Installe les libs Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout ton code
COPY . .

# Expose le port 8000
EXPOSE 8000

# Commande de démarrage (Lance FastAPI)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
