"""описывает основные настройки проекта:
загружает конфигурацию из .env,
хранит параметры для qdrant, s3/minio, llm, embeddings, telegram-бота и rag,
а также предоставляет единый доступ к настройкам через get_settings()."""


from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Qdrant ---
    QDRANT_URL: str
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "chunks_ru"

    # --- MinIO (S3) ---
    S3_ENDPOINT: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_REGION: str = "us-east-1"
    S3_BUCKET_SOURCES: str = "sources"
    S3_BUCKET_ARTIFACTS: str = "artifacts"
    S3_PUBLIC_BASE_URL: str = ""

    # --- LM Studio ---
    LLM_BASE_URL: str
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "local-model"

    # --- Embeddings ---
    EMBED_BASE_URL: str
    EMBED_API_KEY: str = "lm-studio"
    EMBED_MODEL: str
    EMBED_DIM: int = 768

    # --- App ---
    APP_ENV: str = "local"
    RAG_MODE: str = "naive"
    TELEGRAM_BOT_TOKEN: str = ""
    BOT_TOP_K: int = 5

    # --- Yandex Object Storage source links ---
    YANDEX_SOURCE_LINKS_ENABLED: bool = False
    YANDEX_STORAGE_ENDPOINT: str = "https://storage.yandexcloud.net"
    YANDEX_STORAGE_BUCKET: str = ""
    YANDEX_STORAGE_ACCESS_KEY: str = ""
    YANDEX_STORAGE_SECRET_KEY: str = ""
    YANDEX_STORAGE_REGION: str = ""
    YANDEX_SOURCE_LINKS_USE_FILENAME_AS_KEY: bool = True
    YANDEX_SOURCE_LINKS_EXPIRES_SEC: int = 86400

    # Флаг строгости проверки ссылок в ответе
    # мягким режимом — ответ можно оставить;
    # строгим режимом — без нормальных citations лучше отказ.

    STRICT_CITATIONS: bool = False



def get_settings() -> Settings:
    return Settings()