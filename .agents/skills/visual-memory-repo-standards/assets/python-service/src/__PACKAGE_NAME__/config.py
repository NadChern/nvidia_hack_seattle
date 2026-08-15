from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VMA_",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "__SERVICE_NAME__"


@lru_cache
def get_settings() -> Settings:
    return Settings()
