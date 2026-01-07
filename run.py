import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    # C'est ici qu'on configure le serveur (Host 0.0.0.0 pour Docker/VPS)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Met False en production
    )
