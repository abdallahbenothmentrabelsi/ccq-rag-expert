from vector_db import QdrantStorage

# On se connecte
store = QdrantStorage()

# On demande les 3 premiers points de la collection
# (scroll permet de lister le contenu sans faire de recherche sémantique)
res = store.client.scroll(
    collection_name="docs",
    limit=3,
    with_payload=True,
    with_vectors=False # On ne veut pas voir les chiffres, juste le texte
)

print(f"J'ai trouvé {len(res[0])} morceaux de texte :\n")

for point in res[0]:
    print(f"--- SOURCE: {point.payload['source']} ---")
    # On affiche juste les 100 premiers caractères du texte
    print(f"TEXTE: {point.payload['text'][:100]}...\n")
