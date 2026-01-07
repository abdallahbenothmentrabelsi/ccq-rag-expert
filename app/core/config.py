import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "CCQ RAG Expert"
    OPENAI_API_KEY: str
    LLAMA_CLOUD_API_KEY: str
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    INNGEST_EVENT_KEY: str = "local_dev_key"
    INNGEST_IS_PROD: bool = False
    INNGEST_BASE_URL: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
