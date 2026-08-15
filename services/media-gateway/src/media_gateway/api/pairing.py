"""Short-lived QR pairing exchange for device-scoped credentials."""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field

from media_gateway.deps import authorize_request
from media_gateway.domain.pairing import PairingRegistry
from media_gateway.domain.ratelimit import FixedWindowLimiter
from media_gateway.errors import CapacityError, ForbiddenError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pairing"], prefix="/v1/pairing")


class PairingCodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairing_code: str
    expires_at: dt.datetime


class PairingClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairing_code: str = Field(min_length=16, max_length=256)
    device_id: str = Field(min_length=1, max_length=128)


class DeviceCredentialResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: str
    credential: str
    expires_at: dt.datetime


def _client_of(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("", response_model=PairingCodeResponse, status_code=status.HTTP_201_CREATED)
def issue_pairing_code(request: Request) -> PairingCodeResponse:
    authorize_request(request)
    registry: PairingRegistry = request.app.state.pairing
    issued = registry.issue()
    logger.info("issued a pairing code", extra={"expires_at": issued.expires_at})
    return PairingCodeResponse(pairing_code=issued.code, expires_at=issued.expires_at)


@router.post("/claim", response_model=DeviceCredentialResponse)
def claim_pairing_code(request: Request, body: PairingClaimRequest) -> DeviceCredentialResponse:
    limiter: FixedWindowLimiter = request.app.state.pairing_claim_limiter
    client = _client_of(request)
    if not limiter.allow(client):
        raise CapacityError(
            "too many pairing claims",
            retry_after_s=round(limiter.retry_after_s(client), 1),
        )

    allowlist = request.app.state.settings.device_id_allowlist
    if allowlist and body.device_id not in allowlist:
        raise ForbiddenError("device is not authorized", device_id=body.device_id)

    registry: PairingRegistry = request.app.state.pairing
    issued = registry.claim(code=body.pairing_code, device_id=body.device_id)
    logger.info("paired a device", extra={"device_id": body.device_id})
    return DeviceCredentialResponse(
        device_id=issued.device_id,
        credential=issued.credential,
        expires_at=issued.expires_at,
    )


__all__ = [
    "DeviceCredentialResponse",
    "PairingClaimRequest",
    "PairingCodeResponse",
    "router",
]
