from fastapi import FastAPI
from pydantic import BaseModel

from __PACKAGE_NAME__.config import get_settings


class HealthResponse(BaseModel):
    status: str
    service: str


settings = get_settings()
app = FastAPI(title=settings.service_name)


@app.get("/health/live", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name)


@app.get("/health/ready", response_model=HealthResponse)
def readiness() -> HealthResponse:
    return HealthResponse(status="ready", service=settings.service_name)
