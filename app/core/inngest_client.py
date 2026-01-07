import logging
import inngest
from app.core.config import settings

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=settings.INNGEST_IS_PROD,
    event_key=settings.INNGEST_EVENT_KEY,
)
