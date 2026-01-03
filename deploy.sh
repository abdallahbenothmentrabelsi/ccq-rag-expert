#!/bin/bash

echo " Démarrage du déploiement..."

# 1. Aller dans le dossier du projet
cd /root/ccq-rag-expert  # <--- REMPLACE par le VRAI chemin de ton dossier sur le VPS

# 2. Récupérer le dernier code depuis GitHub
echo " Pulling latest code..."
git pull origin main

# 3. Reconstruire et relancer les conteneurs (Seulement ceux qui ont changé)
echo " Rebuilding containers..."
docker compose up -d --build --remove-orphans

# 4. Nettoyer les vieilles images inutiles (Pour ne pas remplir le disque)
echo " Cleaning up..."
docker image prune -f

echo " Déploiement terminé avec succès !"
