import os
# Colle EXACTEMENT le chemin que tu envoies à Inngest ici
path = "C:/Users/Abdo/Desktop/NodeLuck/rag/PycharmProjects/uploads/CCQ-2010.pdf"

if os.path.exists(path):
    print("✅ Le fichier existe ! Python le voit.")
else:
    print("❌ Fichier INTROUVABLE. Vérifie le chemin.")
